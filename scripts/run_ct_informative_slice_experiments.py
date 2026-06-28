#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / "results" / ".matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import config
from src.data.ct_preprocessing import get_ct_dataframes
from src.data.ct_slice_selection import SLICE_SELECTION_VARIANTS, prepare_selection_metadata
from src.data.datasets import CTDataset
from src.data.transforms import get_ct_transforms
from src.training.classification_experiment import (
    get_device,
    make_run_config,
    save_classification_artifacts,
    seed_everything,
    train_and_evaluate,
)


CT_LABEL_MAP = {"CT-0": 0, "CT-1": 1, "CT-2": 2, "CT-3+": 3}
DEFAULT_EXPERIMENTS = (
    ("central30_70", "resnet50", "weighted_ce"),
    ("central30_70", "densenet121", "baseline"),
    ("top16_tissue", "resnet50", "weighted_ce"),
    ("top16_tissue", "densenet121", "baseline"),
    ("top20_tissue", "resnet50", "weighted_ce"),
    ("top20_tissue", "densenet121", "baseline"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare and train CT informative-slice classification experiments.",
    )
    parser.add_argument(
        "command",
        choices=("prepare", "train-default", "train-one"),
        help="Use prepare to only create filtered metadata; train-default runs the planned CT variants.",
    )
    parser.add_argument(
        "--variant",
        choices=tuple(SLICE_SELECTION_VARIANTS),
        help="Variant to train when command=train-one.",
    )
    parser.add_argument(
        "--architecture",
        choices=("resnet50", "densenet121", "efficientnet_b0"),
        help="Architecture to train when command=train-one.",
    )
    parser.add_argument(
        "--strategy",
        choices=("baseline", "weighted_ce", "focal_loss", "oversampling"),
        help="Balance strategy to train when command=train-one.",
    )
    parser.add_argument(
        "--force-quality",
        action="store_true",
        help="Recompute slice quality features even if the cache already exists.",
    )
    return parser.parse_args()


def selection_dir() -> Path:
    return config.CT_DIR / "processed_2d_slices" / "informative_slice_metadata"


def plot_selection_summary(summary: pd.DataFrame, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "ct_slice_selection_keep_ratio.png"
    plot_df = summary.copy()
    plot_df["slice_keep_pct"] = 100 * plot_df["slice_keep_ratio"]

    fig, ax = plt.subplots(figsize=(10, 5), dpi=160)
    for label, group in plot_df.groupby("label"):
        ax.plot(group["variant"], group["slice_keep_pct"], marker="o", label=label)
    ax.set_ylabel("% de slices conservados")
    ax.set_xlabel("Variante")
    ax.set_title("CT: reduccion de slices por variante y clase")
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(title="Clase")
    plt.xticks(rotation=25, ha="right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def prepare_metadata(force_quality: bool = False) -> dict[str, Path]:
    metadata_path = config.CT_DIR / "processed_2d_slices" / "labels_metadata.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing CT metadata: {metadata_path}")

    summary, metadata_paths = prepare_selection_metadata(
        metadata_path=metadata_path,
        output_dir=selection_dir(),
        force_quality=force_quality,
    )
    results_dir = PROJECT_ROOT / "results" / "classification" / "ct_informative_slices"
    results_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(results_dir / "ct_slice_selection_summary.csv", index=False)
    figure_path = plot_selection_summary(summary, results_dir / "figures")
    print("\nSlice selection summary:")
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
    print(f"\nSaved figure: {figure_path}")
    return metadata_paths


def run_experiment(variant_name: str, architecture: str, strategy: str, device: str) -> dict:
    metadata_path = selection_dir() / f"{variant_name}_metadata.csv"
    if not metadata_path.exists():
        prepare_metadata()
    metadata_df = pd.read_csv(metadata_path)
    train_df, val_df, test_df = get_ct_dataframes(metadata_df, config.RANDOM_SEED)

    dataset_name = f"ct_{variant_name}"
    run_config = make_run_config(dataset_name, architecture, strategy, "full")
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
            "slice_selection_variant": variant_name,
            "slice_selection_description": SLICE_SELECTION_VARIANTS[variant_name].description,
            "metadata_path": str(metadata_path),
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


def validate_train_one_args(args: argparse.Namespace) -> None:
    missing = [name for name in ("variant", "architecture", "strategy") if getattr(args, name) is None]
    if missing:
        raise ValueError(f"train-one requires: {', '.join('--' + name for name in missing)}")


def main() -> None:
    args = parse_args()
    seed_everything(config.RANDOM_SEED)

    if args.command == "prepare":
        prepare_metadata(force_quality=args.force_quality)
        return

    prepare_metadata(force_quality=args.force_quality)
    device = get_device()
    print(f"device={device}")

    if args.command == "train-one":
        validate_train_one_args(args)
        summaries = [run_experiment(args.variant, args.architecture, args.strategy, device)]
    else:
        summaries = [
            run_experiment(variant, architecture, strategy, device)
            for variant, architecture, strategy in DEFAULT_EXPERIMENTS
        ]

    summary_df = pd.DataFrame(summaries).sort_values(["accuracy", "f1_macro"], ascending=False)
    print("\nCompleted informative-slice experiments:")
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
