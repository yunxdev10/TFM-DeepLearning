import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, Type

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.config import config
from src.data.datasets import get_balanced_sampler
from src.evaluation.metrics import compute_classification_metrics, predict_classification
from src.models.classifiers import CovidClassifier
from src.training.losses import FocalLoss, get_class_weights
from src.training.trainer import Trainer


BALANCE_STRATEGIES = ("baseline", "weighted_ce", "focal_loss", "oversampling")


@dataclass(frozen=True)
class ExperimentRunConfig:
    dataset_name: str
    architecture: str
    balance_strategy: str
    run_mode: str = "full"
    batch_size: int = config.BATCH_SIZE
    epochs: int = config.EPOCHS
    pretrained: bool = True
    train_limit_per_class: Optional[int] = None
    eval_limit_per_class: Optional[int] = None
    head_epochs: int = 5
    fine_tune_epochs: int = 15

    @property
    def experiment_name(self) -> str:
        return f"{self.dataset_name}_{self.architecture}_{self.balance_strategy}"


def seed_everything(seed: int = config.RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def make_run_config(
    dataset_name: str,
    architecture: str,
    balance_strategy: str,
    run_mode: str = "full",
) -> ExperimentRunConfig:
    if balance_strategy not in BALANCE_STRATEGIES:
        raise ValueError(f"Unknown balance_strategy={balance_strategy!r}. Expected one of {BALANCE_STRATEGIES}.")

    if run_mode != "full":
        raise ValueError("Only run_mode='full' is supported for TFM experiments.")

    return ExperimentRunConfig(
        dataset_name=dataset_name,
        architecture=architecture,
        balance_strategy=balance_strategy,
        run_mode=run_mode,
        batch_size=config.BATCH_SIZE,
        epochs=config.EPOCHS,
        pretrained=True,
        train_limit_per_class=None,
        eval_limit_per_class=None,
        head_epochs=5,
        fine_tune_epochs=max(config.EPOCHS - 5, 0),
    )


def limit_per_class(df: pd.DataFrame, limit: Optional[int], seed: int = config.RANDOM_SEED) -> pd.DataFrame:
    if limit is None:
        return df.reset_index(drop=True)
    return (
        df.groupby("label", group_keys=False)
        .apply(lambda group: group.sample(n=min(limit, len(group)), random_state=seed))
        .reset_index(drop=True)
    )


def label_indices(df: pd.DataFrame, label_map: Dict[str, int]) -> list[int]:
    return [label_map[label] for label in df["label"].tolist()]


def class_counts(df: pd.DataFrame, label_map: Dict[str, int]) -> list[int]:
    labels = label_indices(df, label_map)
    return np.bincount(labels, minlength=len(label_map)).tolist()


def build_criterion(balance_strategy: str, train_df: pd.DataFrame, label_map: Dict[str, int], device: str) -> nn.Module:
    counts = class_counts(train_df, label_map)
    weights = get_class_weights(counts, len(train_df)).to(device)

    if balance_strategy == "baseline" or balance_strategy == "oversampling":
        return nn.CrossEntropyLoss()
    if balance_strategy == "weighted_ce":
        return nn.CrossEntropyLoss(weight=weights)
    if balance_strategy == "focal_loss":
        return FocalLoss(alpha=weights, gamma=2.0)
    raise ValueError(f"Unknown balance_strategy={balance_strategy!r}.")


def _build_trainer(
    model: nn.Module,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
) -> Trainer:
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
    return Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        scheduler=scheduler,
        early_stopping_patience=config.EARLY_STOPPING_PATIENCE,
    )


def _merge_history(base: Optional[Dict[str, list]], new: Dict[str, list], phase: str) -> Dict[str, list]:
    if base is None:
        base = {key: [] for key in new}
        base["phase"] = []
    for key, values in new.items():
        base.setdefault(key, []).extend(values)
    base.setdefault("phase", []).extend([phase] * len(next(iter(new.values()), [])))
    return base


def build_loaders(
    dataset_cls: Type[Dataset],
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_transform: Callable,
    eval_transform: Callable,
    label_map: Dict[str, int],
    batch_size: int,
    balance_strategy: str,
    num_workers: int,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_dataset = dataset_cls(train_df, transform=train_transform, label_map=label_map)
    val_dataset = dataset_cls(val_df, transform=eval_transform, label_map=label_map)
    test_dataset = dataset_cls(test_df, transform=eval_transform, label_map=label_map)

    sampler = None
    shuffle = True
    if balance_strategy == "oversampling":
        sampler = get_balanced_sampler(label_indices(train_df, label_map))
        shuffle = False

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader


def train_and_evaluate(
    run_config: ExperimentRunConfig,
    dataset_cls: Type[Dataset],
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_transform: Callable,
    eval_transform: Callable,
    label_map: Dict[str, int],
    in_channels: int,
    device: str,
) -> Dict[str, Any]:
    seed_everything(config.RANDOM_SEED)
    num_workers = config.NUM_WORKERS
    train_run_df = limit_per_class(train_df, run_config.train_limit_per_class)
    val_run_df = limit_per_class(val_df, run_config.eval_limit_per_class)
    test_run_df = limit_per_class(test_df, run_config.eval_limit_per_class)

    train_loader, val_loader, test_loader = build_loaders(
        dataset_cls=dataset_cls,
        train_df=train_run_df,
        val_df=val_run_df,
        test_df=test_run_df,
        train_transform=train_transform,
        eval_transform=eval_transform,
        label_map=label_map,
        batch_size=run_config.batch_size,
        balance_strategy=run_config.balance_strategy,
        num_workers=num_workers,
    )

    model = CovidClassifier(
        architecture_name=run_config.architecture,
        num_classes=len(label_map),
        in_channels=in_channels,
        pretrained=run_config.pretrained,
    )
    criterion = build_criterion(run_config.balance_strategy, train_run_df, label_map, device)
    history = None

    if run_config.pretrained and run_config.head_epochs > 0:
        model.freeze_backbone()
        head_optimizer = torch.optim.AdamW(
            model.classifier_parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY,
        )
        head_trainer = _build_trainer(model, criterion, head_optimizer, device)
        model = head_trainer.fit(train_loader, val_loader, epochs=run_config.head_epochs)
        history = _merge_history(history, head_trainer.history, "head")

        if run_config.fine_tune_epochs > 0:
            model.unfreeze_all()
            fine_tune_optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=config.LEARNING_RATE * 0.1,
                weight_decay=config.WEIGHT_DECAY,
            )
            fine_tune_trainer = _build_trainer(model, criterion, fine_tune_optimizer, device)
            model = fine_tune_trainer.fit(train_loader, val_loader, epochs=run_config.fine_tune_epochs)
            history = _merge_history(history, fine_tune_trainer.history, "fine_tune")
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
        trainer = _build_trainer(model, criterion, optimizer, device)
        model = trainer.fit(train_loader, val_loader, epochs=run_config.epochs)
        history = _merge_history(history, trainer.history, "single_stage")

    predictions = predict_classification(model, test_loader, device)
    metrics = compute_classification_metrics(
        predictions["y_true"],
        predictions["y_pred"],
        predictions["y_prob"],
    )

    return {
        "model": model,
        "history": history,
        "metrics": metrics,
        "predictions": predictions,
        "split_summary": pd.concat(
            [
                train_run_df.assign(split="train"),
                val_run_df.assign(split="val"),
                test_run_df.assign(split="test"),
            ]
        )
        .groupby(["split", "label"])
        .size()
        .unstack(fill_value=0),
        "train_df": train_run_df,
        "val_df": val_run_df,
        "test_df": test_run_df,
    }


def save_classification_artifacts(
    run_config: ExperimentRunConfig,
    result: Dict[str, Any],
    label_map: Dict[str, int],
    models_dir: Path,
    results_dir: Path,
) -> Dict[str, Path]:
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    artifact_name = f"{run_config.experiment_name}_{run_config.run_mode}"
    model_path = models_dir / f"{artifact_name}.pt"
    history_path = results_dir / f"{artifact_name}_history.csv"
    summary_path = results_dir / f"{artifact_name}_summary.json"
    report_path = results_dir / f"{artifact_name}_classification_report.csv"
    confusion_path = results_dir / f"{artifact_name}_confusion_matrix.csv"
    predictions_path = results_dir / f"{artifact_name}_predictions.csv"

    metrics = result["metrics"]
    summary = {
        "experiment": run_config.experiment_name,
        "dataset": run_config.dataset_name,
        "run_mode": run_config.run_mode,
        "architecture": run_config.architecture,
        "balance_strategy": run_config.balance_strategy,
        "accuracy": metrics["accuracy"],
        "f1_macro": metrics["f1_macro"],
        "f1_weighted": metrics["f1_weighted"],
        "auc_roc_macro": metrics.get("auc_roc_macro"),
    }

    torch.save(
        {
            "model_state_dict": result["model"].state_dict(),
            "architecture": run_config.architecture,
            "label_map": label_map,
            "run_mode": run_config.run_mode,
            "balance_strategy": run_config.balance_strategy,
            "history": result["history"],
        },
        model_path,
    )
    pd.DataFrame(result["history"]).to_csv(history_path, index=False)
    pd.DataFrame(metrics["classification_report"]).T.to_csv(report_path)
    pd.DataFrame(metrics["confusion_matrix"]).to_csv(confusion_path, index=False)
    predictions_df = pd.DataFrame(
        {
            "y_true": result["predictions"]["y_true"],
            "y_pred": result["predictions"]["y_pred"],
        }
    )
    for class_idx in range(result["predictions"]["y_prob"].shape[1]):
        predictions_df[f"prob_{class_idx}"] = result["predictions"]["y_prob"][:, class_idx]
    predictions_df.to_csv(predictions_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2))

    return {
        "model": model_path,
        "history": history_path,
        "summary": summary_path,
        "classification_report": report_path,
        "confusion_matrix": confusion_path,
        "predictions": predictions_path,
    }
