from pathlib import Path
import json
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.config import config
from src.data.ct_preprocessing import get_ct_dataframes
from src.data.datasets import CTDataset, CXRDataset
from src.data.preprocessing import get_cxr_dataframes
from src.data.transforms import get_ct_transforms, get_cxr_transforms
from src.training.classification_experiment import (
    BALANCE_STRATEGIES,
    get_device,
    make_run_config,
    save_classification_artifacts,
    seed_everything,
    train_and_evaluate,
)


ARCHITECTURES = ("resnet50", "densenet121", "efficientnet_b0")
CXR_LABEL_MAP = {"COVID": 0, "Lung_Opacity": 1, "Normal": 2, "Viral Pneumonia": 3}
CT_LABEL_MAP = {"CT-0": 0, "CT-1": 1, "CT-2": 2, "CT-3+": 3}


def run_matrix(
    dataset_name: str,
    dataset_cls,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_transform,
    eval_transform,
    label_map: dict,
    in_channels: int,
    device: str,
) -> list[dict]:
    models_dir = config.MODELS_DIR / dataset_name
    results_dir = PROJECT_ROOT / "results" / "classification" / dataset_name
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for architecture in ARCHITECTURES:
        for strategy in BALANCE_STRATEGIES:
            run_config = make_run_config(dataset_name, architecture, strategy, "full")
            artifact_name = f"{run_config.experiment_name}_{run_config.run_mode}"
            model_path = models_dir / f"{artifact_name}.pt"
            summary_path = results_dir / f"{artifact_name}_summary.json"
            print(f"\n=== {artifact_name} ===")

            if model_path.exists() and summary_path.exists():
                print("Saltado: artefactos full existentes detectados.")
                summaries.append(json.loads(summary_path.read_text()))
                continue

            result = train_and_evaluate(
                run_config=run_config,
                dataset_cls=dataset_cls,
                train_df=train_df,
                val_df=val_df,
                test_df=test_df,
                train_transform=train_transform,
                eval_transform=eval_transform,
                label_map=label_map,
                in_channels=in_channels,
                device=device,
            )
            paths = save_classification_artifacts(run_config, result, label_map, models_dir, results_dir)
            summaries.append(json.loads(paths["summary"].read_text()))
    return summaries


def main() -> None:
    seed_everything(config.RANDOM_SEED)
    device = get_device()
    print(f"device={device}")
    print("run_mode=full")

    cxr_train, cxr_val, cxr_test = get_cxr_dataframes(config.CXR_DIR, config.RANDOM_SEED)
    ct_metadata_path = config.CT_DIR / "processed_2d_slices" / "labels_metadata.csv"
    if not ct_metadata_path.exists():
        raise FileNotFoundError(f"Missing CT metadata at {ct_metadata_path}. Run CT preprocessing first.")
    ct_df = pd.read_csv(ct_metadata_path)
    ct_train, ct_val, ct_test = get_ct_dataframes(ct_df, config.RANDOM_SEED)

    summaries = []
    summaries.extend(
        run_matrix(
            dataset_name="cxr",
            dataset_cls=CXRDataset,
            train_df=cxr_train,
            val_df=cxr_val,
            test_df=cxr_test,
            train_transform=get_cxr_transforms(config.CXR_IMAGE_SIZE, is_train=True),
            eval_transform=get_cxr_transforms(config.CXR_IMAGE_SIZE, is_train=False),
            label_map=CXR_LABEL_MAP,
            in_channels=3,
            device=device,
        )
    )
    summaries.extend(
        run_matrix(
            dataset_name="ct",
            dataset_cls=CTDataset,
            train_df=ct_train,
            val_df=ct_val,
            test_df=ct_test,
            train_transform=get_ct_transforms(config.CT_IMAGE_SIZE, is_train=True),
            eval_transform=get_ct_transforms(config.CT_IMAGE_SIZE, is_train=False),
            label_map=CT_LABEL_MAP,
            in_channels=1,
            device=device,
        )
    )

    summary_df = pd.DataFrame(summaries).sort_values(["dataset", "accuracy", "f1_macro"], ascending=[True, False, False])
    print(summary_df[["dataset", "experiment", "architecture", "balance_strategy", "accuracy", "f1_macro"]])


if __name__ == "__main__":
    main()
