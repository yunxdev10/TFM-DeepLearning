#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / "results" / ".matplotlib"))

from src.config import config
from src.data.ct_preprocessing import get_ct_dataframes
from src.data.ct_slice_selection import SLICE_SELECTION_VARIANTS, prepare_selection_metadata
from src.data.datasets import CTDataset
from src.data.transforms import get_ct_transforms
from src.training.classification_experiment import (
    ExperimentRunConfig,
    get_device,
    make_run_config,
    save_classification_artifacts,
    seed_everything,
    train_and_evaluate,
)


CT_LABEL_MAP = {"CT-0": 0, "CT-1": 1, "CT-2": 2, "CT-3+": 3}
MODERN_ARCHITECTURES = ("convnext_tiny", "efficientnet_v2_s")
DEFAULT_EXPERIMENTS = (
    ("top20_tissue", "convnext_tiny", "weighted_ce"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train modern backbone CT classification experiments on informative slices.",
    )
    parser.add_argument(
        "command",
        choices=("prepare", "train-default", "train-one"),
        help="prepare creates metadata only; train-default runs the planned modern CT experiment.",
    )
    parser.add_argument(
        "--variant",
        default="top20_tissue",
        choices=tuple(SLICE_SELECTION_VARIANTS),
        help="Informative-slice variant used for training.",
    )
    parser.add_argument(
        "--architecture",
        default="convnext_tiny",
        choices=MODERN_ARCHITECTURES,
        help="Modern backbone to train.",
    )
    parser.add_argument(
        "--strategy",
        default="weighted_ce",
        choices=("baseline", "weighted_ce", "focal_loss", "oversampling"),
        help="Class-imbalance strategy.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=config.EPOCHS,
        help="Total epochs. The first --head-epochs train only the classifier head.",
    )
    parser.add_argument(
        "--head-epochs",
        type=int,
        default=5,
        help="Epochs with frozen backbone before fine-tuning.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size. Modern backbones are heavier than ResNet50/DenseNet121.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader workers. Use 0 for maximum compatibility in notebooks/macOS.",
    )
    parser.add_argument(
        "--force-quality",
        action="store_true",
        help="Recompute slice quality features even if the cache already exists.",
    )
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Train without ImageNet pretrained weights if weight download is unavailable.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=config.LEARNING_RATE,
        help="Learning rate used for the classifier head. Fine-tuning uses 0.1x this value.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=config.WEIGHT_DECAY,
        help="AdamW weight decay.",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=config.EARLY_STOPPING_PATIENCE,
        help="Early stopping patience based on validation loss.",
    )
    parser.add_argument(
        "--tag",
        default="",
        help="Optional experiment tag, e.g. head_only or gentle_ft, to avoid overwriting previous runs.",
    )
    return parser.parse_args()


def selection_dir() -> Path:
    return config.CT_DIR / "processed_2d_slices" / "informative_slice_metadata"


def prepare_metadata(force_quality: bool = False, variants: list[str] | None = None) -> dict[str, Path]:
    metadata_path = config.CT_DIR / "processed_2d_slices" / "labels_metadata.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing CT metadata: {metadata_path}")

    summary, metadata_paths = prepare_selection_metadata(
        metadata_path=metadata_path,
        output_dir=selection_dir(),
        variants=variants or [variant for variant, _, _ in DEFAULT_EXPERIMENTS],
        force_quality=force_quality,
    )
    results_dir = PROJECT_ROOT / "results" / "classification" / "ct_informative_slices"
    results_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(results_dir / "ct_modern_backbone_slice_selection_summary.csv", index=False)
    print("\nSlice selection used by modern-backbone experiments:")
    print(
        summary[
            [
                "variant",
                "label",
                "base_slices",
                "selected_slices",
                "slice_keep_ratio",
                "base_studies",
                "selected_studies",
            ]
        ].round(4)
    )
    return metadata_paths


def make_modern_run_config(
    dataset_name: str,
    architecture: str,
    strategy: str,
    epochs: int,
    head_epochs: int,
    batch_size: int,
    pretrained: bool,
) -> ExperimentRunConfig:
    base_config = make_run_config(dataset_name, architecture, strategy, "full")
    head_epochs = max(min(head_epochs, epochs), 0)
    return replace(
        base_config,
        epochs=epochs,
        batch_size=batch_size,
        head_epochs=head_epochs,
        fine_tune_epochs=max(epochs - head_epochs, 0),
        pretrained=pretrained,
    )


def run_experiment(
    variant_name: str,
    architecture: str,
    strategy: str,
    epochs: int,
    head_epochs: int,
    batch_size: int,
    pretrained: bool,
    tag: str,
    device: str,
) -> dict:
    metadata_path = selection_dir() / f"{variant_name}_metadata.csv"
    if not metadata_path.exists():
        prepare_metadata()
    metadata_df = pd.read_csv(metadata_path)
    train_df, val_df, test_df = get_ct_dataframes(metadata_df, config.RANDOM_SEED)

    dataset_name = f"ct_{variant_name}" if not tag else f"ct_{variant_name}_{tag}"
    run_config = make_modern_run_config(
        dataset_name,
        architecture,
        strategy,
        epochs,
        head_epochs,
        batch_size,
        pretrained,
    )
    artifact_name = f"{run_config.experiment_name}_{run_config.run_mode}"
    models_dir = config.MODELS_DIR / "ct_informative_slices"
    results_dir = PROJECT_ROOT / "results" / "classification" / "ct_informative_slices"
    model_path = models_dir / f"{artifact_name}.pt"
    summary_path = results_dir / f"{artifact_name}_summary.json"

    print(f"\n=== {artifact_name} ===")
    if model_path.exists() and summary_path.exists():
        print("Saltado: artefactos full existentes detectados.")
        return json.loads(summary_path.read_text())

    result = train_and_evaluate(
        run_config=run_config,
        dataset_cls=CTDataset,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        train_transform=get_ct_transforms(config.CT_IMAGE_SIZE, is_train=True),
        eval_transform=get_ct_transforms(config.CT_IMAGE_SIZE, is_train=False),
        label_map=CT_LABEL_MAP,
        in_channels=1,
        device=device,
    )
    paths = save_classification_artifacts(run_config, result, CT_LABEL_MAP, models_dir, results_dir)

    summary = json.loads(paths["summary"].read_text())
    summary.update(
        {
            "base_dataset": "ct",
            "experiment_family": "ct_modern_backbone",
            "slice_selection_variant": variant_name,
            "slice_selection_description": SLICE_SELECTION_VARIANTS[variant_name].description,
            "metadata_path": str(metadata_path),
            "hyperparameters": {
                "epochs": epochs,
                "head_epochs": run_config.head_epochs,
                "fine_tune_epochs": run_config.fine_tune_epochs,
                "batch_size": batch_size,
                "learning_rate": config.LEARNING_RATE,
                "fine_tune_learning_rate": config.LEARNING_RATE * 0.1,
                "weight_decay": config.WEIGHT_DECAY,
                "early_stopping_patience": config.EARLY_STOPPING_PATIENCE,
                "pretrained": run_config.pretrained,
                "tag": tag,
            },
            "split_sizes": {
                "train": int(len(train_df)),
                "val": int(len(val_df)),
                "test": int(len(test_df)),
                "train_studies": int(train_df["study_id"].nunique()),
                "val_studies": int(val_df["study_id"].nunique()),
                "test_studies": int(test_df["study_id"].nunique()),
            },
        }
    )
    paths["summary"].write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    args = parse_args()
    seed_everything(config.RANDOM_SEED)

    if args.command == "prepare":
        prepare_metadata(force_quality=args.force_quality)
        return

    config.NUM_WORKERS = args.num_workers
    config.LEARNING_RATE = args.learning_rate
    config.WEIGHT_DECAY = args.weight_decay
    config.EARLY_STOPPING_PATIENCE = args.early_stopping_patience

    variants_to_prepare = [args.variant] if args.command == "train-one" else [variant for variant, _, _ in DEFAULT_EXPERIMENTS]
    prepare_metadata(force_quality=args.force_quality, variants=variants_to_prepare)
    device = get_device()
    print(f"device={device}")

    if args.command == "train-one":
        experiments = ((args.variant, args.architecture, args.strategy),)
    else:
        experiments = DEFAULT_EXPERIMENTS

    summaries = [
        run_experiment(
            variant_name=variant,
            architecture=architecture,
            strategy=strategy,
            epochs=args.epochs,
            head_epochs=args.head_epochs,
            batch_size=args.batch_size,
            pretrained=not args.no_pretrained,
            tag=args.tag,
            device=device,
        )
        for variant, architecture, strategy in experiments
    ]

    summary_df = pd.DataFrame(summaries).sort_values(["accuracy", "f1_macro"], ascending=False)
    print("\nCompleted modern-backbone CT experiments:")
    print(
        summary_df[
            [
                "experiment",
                "slice_selection_variant",
                "architecture",
                "balance_strategy",
                "accuracy",
                "f1_macro",
                "f1_weighted",
                "auc_roc_macro",
            ]
        ]
    )


if __name__ == "__main__":
    main()
