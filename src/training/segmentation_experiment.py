from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import json
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import config
from src.data.segmentation import SegmentationDataset, SegmentationPairTransform
from src.models.segmentation import build_segmentation_model


SEGMENTATION_ARCHITECTURES = ["unet", "attention_unet"]


@dataclass(frozen=True)
class SegmentationRunConfig:
    dataset_name: str
    architecture: str
    run_mode: str
    image_size: Tuple[int, int]
    in_channels: int
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    early_stopping_patience: int
    base_features: int = 32
    variant_name: str = "baseline"
    loss_name: str = "dice_bce"
    bce_weight: float = 0.5
    dice_weight: float = 0.5
    pos_weight: Optional[float] = None
    tversky_alpha: float = 0.3
    tversky_beta: float = 0.7
    threshold: float = 0.5
    optimize_threshold: bool = False
    threshold_search_min: float = 0.2
    threshold_search_max: float = 0.95
    threshold_search_step: float = 0.05
    train_crop_size: Optional[Tuple[int, int]] = None
    train_crop_prob: float = 1.0
    positive_crop_prob: float = 0.0
    ct_context_slices: bool = False

    @property
    def experiment_name(self) -> str:
        if self.variant_name == "baseline":
            return f"{self.dataset_name}_{self.architecture}_segmentation"
        return f"{self.dataset_name}_{self.architecture}_{self.variant_name}_segmentation"


class DiceBCELoss(nn.Module):
    def __init__(
        self,
        smooth: float = 1.0,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
        pos_weight: Optional[float] = None,
    ):
        super().__init__()
        self.smooth = smooth
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.pos_weight = pos_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        pos_weight = None
        if self.pos_weight is not None:
            pos_weight = torch.tensor([self.pos_weight], dtype=logits.dtype, device=logits.device)
        bce_loss = nn.functional.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight)
        probs = torch.sigmoid(logits)
        dims = (1, 2, 3)
        intersection = torch.sum(probs * targets, dims)
        union = torch.sum(probs, dims) + torch.sum(targets, dims)
        dice_loss = 1 - torch.mean((2 * intersection + self.smooth) / (union + self.smooth))
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


class TverskyBCELoss(nn.Module):
    def __init__(
        self,
        smooth: float = 1.0,
        bce_weight: float = 0.3,
        tversky_weight: float = 0.7,
        alpha: float = 0.3,
        beta: float = 0.7,
        pos_weight: Optional[float] = None,
    ):
        super().__init__()
        self.smooth = smooth
        self.bce_weight = bce_weight
        self.tversky_weight = tversky_weight
        self.alpha = alpha
        self.beta = beta
        self.pos_weight = pos_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        pos_weight = None
        if self.pos_weight is not None:
            pos_weight = torch.tensor([self.pos_weight], dtype=logits.dtype, device=logits.device)
        bce_loss = nn.functional.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight)
        probs = torch.sigmoid(logits)
        dims = (1, 2, 3)
        true_positive = torch.sum(probs * targets, dims)
        false_positive = torch.sum(probs * (1 - targets), dims)
        false_negative = torch.sum((1 - probs) * targets, dims)
        tversky = (true_positive + self.smooth) / (
            true_positive + self.alpha * false_positive + self.beta * false_negative + self.smooth
        )
        tversky_loss = 1 - torch.mean(tversky)
        return self.bce_weight * bce_loss + self.tversky_weight * tversky_loss


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def make_segmentation_run_config(
    dataset_name: str,
    architecture: str,
    run_mode: str,
    image_size: Tuple[int, int],
    in_channels: int,
    batch_size: Optional[int] = None,
    epochs: Optional[int] = None,
    learning_rate: Optional[float] = None,
    weight_decay: Optional[float] = None,
    early_stopping_patience: Optional[int] = None,
    base_features: Optional[int] = None,
    variant_name: str = "baseline",
    loss_name: str = "dice_bce",
    bce_weight: float = 0.5,
    dice_weight: float = 0.5,
    pos_weight: Optional[float] = None,
    tversky_alpha: float = 0.3,
    tversky_beta: float = 0.7,
    threshold: float = 0.5,
    optimize_threshold: bool = False,
    threshold_search_min: float = 0.2,
    threshold_search_max: float = 0.95,
    threshold_search_step: float = 0.05,
    train_crop_size: Optional[Tuple[int, int]] = None,
    train_crop_prob: float = 1.0,
    positive_crop_prob: float = 0.0,
    ct_context_slices: bool = False,
) -> SegmentationRunConfig:
    if run_mode != "full":
        raise ValueError("Only run_mode='full' is supported for TFM segmentation experiments.")

    return SegmentationRunConfig(
        dataset_name=dataset_name,
        architecture=architecture,
        run_mode=run_mode,
        image_size=image_size,
        in_channels=in_channels,
        epochs=epochs if epochs is not None else config.SEGMENTATION_EPOCHS,
        batch_size=batch_size if batch_size is not None else config.SEGMENTATION_BATCH_SIZE,
        learning_rate=learning_rate if learning_rate is not None else config.SEGMENTATION_LEARNING_RATE,
        weight_decay=weight_decay if weight_decay is not None else config.SEGMENTATION_WEIGHT_DECAY,
        early_stopping_patience=(
            early_stopping_patience
            if early_stopping_patience is not None
            else config.SEGMENTATION_EARLY_STOPPING_PATIENCE
        ),
        base_features=base_features if base_features is not None else 32,
        variant_name=variant_name,
        loss_name=loss_name,
        bce_weight=bce_weight,
        dice_weight=dice_weight,
        pos_weight=pos_weight,
        tversky_alpha=tversky_alpha,
        tversky_beta=tversky_beta,
        threshold=threshold,
        optimize_threshold=optimize_threshold,
        threshold_search_min=threshold_search_min,
        threshold_search_max=threshold_search_max,
        threshold_search_step=threshold_search_step,
        train_crop_size=train_crop_size,
        train_crop_prob=train_crop_prob,
        positive_crop_prob=positive_crop_prob,
        ct_context_slices=ct_context_slices,
    )


def limit_segmentation_samples(df: pd.DataFrame, run_mode: str, max_samples: int = 64) -> pd.DataFrame:
    if run_mode != "full":
        raise ValueError("Only run_mode='full' is supported for TFM segmentation experiments.")
    return df.reset_index(drop=True)


def build_segmentation_loaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    run_config: SegmentationRunConfig,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_transform = SegmentationPairTransform(
        image_size=run_config.image_size,
        in_channels=run_config.in_channels,
        is_train=True,
        train_crop_size=run_config.train_crop_size,
        train_crop_prob=run_config.train_crop_prob,
        positive_crop_prob=run_config.positive_crop_prob,
    )
    eval_transform = SegmentationPairTransform(
        image_size=run_config.image_size,
        in_channels=run_config.in_channels,
        is_train=False,
    )

    train_dataset = SegmentationDataset(
        train_df,
        transform=train_transform,
        in_channels=run_config.in_channels,
        ct_context_slices=run_config.ct_context_slices,
    )
    val_dataset = SegmentationDataset(
        val_df,
        transform=eval_transform,
        in_channels=run_config.in_channels,
        ct_context_slices=run_config.ct_context_slices,
    )
    test_dataset = SegmentationDataset(
        test_df,
        transform=eval_transform,
        in_channels=run_config.in_channels,
        ct_context_slices=run_config.ct_context_slices,
    )

    return (
        DataLoader(
            train_dataset,
            batch_size=run_config.batch_size,
            shuffle=True,
            num_workers=config.NUM_WORKERS,
        ),
        DataLoader(
            val_dataset,
            batch_size=run_config.batch_size,
            shuffle=False,
            num_workers=config.NUM_WORKERS,
        ),
        DataLoader(
            test_dataset,
            batch_size=run_config.batch_size,
            shuffle=False,
            num_workers=config.NUM_WORKERS,
        ),
    )


def estimate_foreground_pos_weight(
    df: pd.DataFrame,
    max_pos_weight: Optional[float] = 50.0,
) -> float:
    positive_pixels = 0
    total_pixels = 0
    from PIL import Image

    for mask_path in df["mask_path"]:
        mask = np.array(Image.open(mask_path).convert("L")) > 0
        positive_pixels += int(mask.sum())
        total_pixels += int(mask.size)
    if positive_pixels == 0:
        return 1.0
    pos_weight = (total_pixels - positive_pixels) / positive_pixels
    if max_pos_weight is not None:
        pos_weight = min(pos_weight, max_pos_weight)
    return float(pos_weight)


def build_segmentation_loss(run_config: SegmentationRunConfig) -> nn.Module:
    loss_name = run_config.loss_name.lower()
    if loss_name in {"dice_bce", "weighted_dice_bce"}:
        return DiceBCELoss(
            bce_weight=run_config.bce_weight,
            dice_weight=run_config.dice_weight,
            pos_weight=run_config.pos_weight,
        )
    if loss_name in {"tversky_bce", "weighted_tversky_bce"}:
        return TverskyBCELoss(
            bce_weight=run_config.bce_weight,
            tversky_weight=run_config.dice_weight,
            alpha=run_config.tversky_alpha,
            beta=run_config.tversky_beta,
            pos_weight=run_config.pos_weight,
        )
    raise ValueError(f"Unsupported segmentation loss: {run_config.loss_name}")


def compute_segmentation_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    smooth: float = 1.0,
) -> Dict[str, float]:
    return compute_segmentation_metrics_from_probs(torch.sigmoid(logits), targets, threshold, smooth)


def compute_segmentation_metrics_from_probs(
    probs: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    smooth: float = 1.0,
) -> Dict[str, float]:
    preds = (probs >= threshold).float()
    dims = (1, 2, 3)
    intersection = torch.sum(preds * targets, dims)
    pred_sum = torch.sum(preds, dims)
    target_sum = torch.sum(targets, dims)
    union = pred_sum + target_sum - intersection

    dice = torch.mean((2 * intersection + smooth) / (pred_sum + target_sum + smooth))
    iou = torch.mean((intersection + smooth) / (union + smooth))
    pixel_accuracy = torch.mean((preds == targets).float())

    return {
        "dice": float(dice.detach().cpu().item()),
        "iou": float(iou.detach().cpu().item()),
        "pixel_accuracy": float(pixel_accuracy.detach().cpu().item()),
    }


class SegmentationTrainer:
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: str,
        early_stopping_patience: int,
    ):
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.early_stopping_patience = early_stopping_patience
        self.history = {
            "train_loss": [],
            "train_dice": [],
            "train_iou": [],
            "val_loss": [],
            "val_dice": [],
            "val_iou": [],
            "val_pixel_accuracy": [],
        }

    def fit(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int) -> nn.Module:
        best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
        best_val_dice = -float("inf")
        epochs_without_improvement = 0

        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            train_metrics = self._run_epoch(train_loader, train=True)
            val_metrics = self._run_epoch(val_loader, train=False)

            self.history["train_loss"].append(train_metrics["loss"])
            self.history["train_dice"].append(train_metrics["dice"])
            self.history["train_iou"].append(train_metrics["iou"])
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["val_dice"].append(val_metrics["dice"])
            self.history["val_iou"].append(val_metrics["iou"])
            self.history["val_pixel_accuracy"].append(val_metrics["pixel_accuracy"])

            print(
                "Train loss={train_loss:.4f} dice={train_dice:.4f} | "
                "Val loss={val_loss:.4f} dice={val_dice:.4f} iou={val_iou:.4f}".format(
                    train_loss=train_metrics["loss"],
                    train_dice=train_metrics["dice"],
                    val_loss=val_metrics["loss"],
                    val_dice=val_metrics["dice"],
                    val_iou=val_metrics["iou"],
                )
            )

            if val_metrics["dice"] > best_val_dice:
                best_val_dice = val_metrics["dice"]
                best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                epochs_without_improvement = 0
                print(">> Saving new best segmentation model.")
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= self.early_stopping_patience:
                    print(">> Early stopping triggered.")
                    break

        self.model.load_state_dict(best_state)
        return self.model

    def _run_epoch(self, dataloader: DataLoader, train: bool) -> Dict[str, float]:
        self.model.train(train)
        total_loss = 0.0
        total_samples = 0
        metric_totals = {"dice": 0.0, "iou": 0.0, "pixel_accuracy": 0.0}

        desc = "Training" if train else "Validation"
        for images, masks in tqdm(dataloader, desc=desc):
            images = images.to(self.device)
            masks = masks.to(self.device)

            if train:
                self.optimizer.zero_grad()

            with torch.set_grad_enabled(train):
                logits = self.model(images)
                loss = self.criterion(logits, masks)
                if train:
                    loss.backward()
                    self.optimizer.step()

            batch_size = images.size(0)
            batch_metrics = compute_segmentation_metrics(logits, masks)
            total_loss += float(loss.detach().cpu().item()) * batch_size
            total_samples += batch_size
            for key in metric_totals:
                metric_totals[key] += batch_metrics[key] * batch_size

        metrics = {key: value / total_samples for key, value in metric_totals.items()}
        metrics["loss"] = total_loss / total_samples
        return metrics


def evaluate_segmentation(
    model: nn.Module,
    dataloader: DataLoader,
    device: str,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    model.eval()
    metric_totals = {"dice": 0.0, "iou": 0.0, "pixel_accuracy": 0.0}
    total_samples = 0
    examples = []

    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc="Testing"):
            images = images.to(device)
            masks = masks.to(device)
            logits = model(images)
            batch_metrics = compute_segmentation_metrics(logits, masks, threshold=threshold)
            batch_size = images.size(0)
            total_samples += batch_size
            for key in metric_totals:
                metric_totals[key] += batch_metrics[key] * batch_size

            if len(examples) < 8:
                probs = torch.sigmoid(logits).detach().cpu()
                for idx in range(min(batch_size, 8 - len(examples))):
                    examples.append(
                        {
                            "image": images[idx].detach().cpu(),
                            "mask": masks[idx].detach().cpu(),
                            "prediction": probs[idx],
                        }
                    )

    metrics = {key: value / total_samples for key, value in metric_totals.items()}
    return {"metrics": metrics, "examples": examples}


def find_best_segmentation_threshold(
    model: nn.Module,
    dataloader: DataLoader,
    device: str,
    thresholds: Optional[Iterable[float]] = None,
) -> float:
    if thresholds is None:
        thresholds = np.arange(0.2, 0.96, 0.05)

    model.eval()
    probs = []
    targets = []
    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc="Threshold tuning"):
            images = images.to(device)
            logits = model(images)
            probs.append(torch.sigmoid(logits).cpu())
            targets.append(masks.cpu())

    probs_tensor = torch.cat(probs)
    targets_tensor = torch.cat(targets)
    best_threshold = 0.5
    best_dice = -float("inf")
    for threshold in thresholds:
        metrics = compute_segmentation_metrics_from_probs(probs_tensor, targets_tensor, threshold=float(threshold))
        if metrics["dice"] > best_dice:
            best_dice = metrics["dice"]
            best_threshold = float(threshold)
    return best_threshold


def train_and_evaluate_segmentation(
    run_config: SegmentationRunConfig,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    device: str,
) -> Dict[str, Any]:
    train_loader, val_loader, test_loader = build_segmentation_loaders(train_df, val_df, test_df, run_config)
    model = build_segmentation_model(
        architecture=run_config.architecture,
        in_channels=run_config.in_channels,
        out_channels=1,
        base_features=run_config.base_features,
    )
    criterion = build_segmentation_loss(run_config)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=run_config.learning_rate,
        weight_decay=run_config.weight_decay,
    )
    trainer = SegmentationTrainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        early_stopping_patience=run_config.early_stopping_patience,
    )
    model = trainer.fit(train_loader, val_loader, epochs=run_config.epochs)
    threshold = run_config.threshold
    if run_config.optimize_threshold:
        thresholds = np.arange(
            run_config.threshold_search_min,
            run_config.threshold_search_max + 1e-9,
            run_config.threshold_search_step,
        )
        threshold = find_best_segmentation_threshold(model, val_loader, device, thresholds=thresholds)
        print(f">> Best validation threshold: {threshold:.2f}")
    evaluation = evaluate_segmentation(model, test_loader, device, threshold=threshold)
    return {
        "model": model,
        "history": trainer.history,
        "metrics": evaluation["metrics"],
        "examples": evaluation["examples"],
        "threshold": threshold,
        "split_sizes": {
            "train": len(train_df),
            "val": len(val_df),
            "test": len(test_df),
        },
    }


def save_segmentation_artifacts(
    run_config: SegmentationRunConfig,
    result: Dict[str, Any],
    models_dir: Path,
    results_dir: Path,
) -> Dict[str, Path]:
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    artifact_name = f"{run_config.experiment_name}_{run_config.run_mode}"
    model_path = models_dir / f"{artifact_name}.pt"
    history_path = results_dir / f"{artifact_name}_history.csv"
    summary_path = results_dir / f"{artifact_name}_summary.json"

    torch.save(
        {
            "model_state_dict": result["model"].state_dict(),
            "architecture": run_config.architecture,
            "dataset_name": run_config.dataset_name,
            "run_mode": run_config.run_mode,
            "image_size": run_config.image_size,
            "in_channels": run_config.in_channels,
            "metrics": result["metrics"],
            "history": result["history"],
            "threshold": result.get("threshold", run_config.threshold),
            "loss_name": run_config.loss_name,
            "variant_name": run_config.variant_name,
            "ct_context_slices": run_config.ct_context_slices,
        },
        model_path,
    )
    pd.DataFrame(result["history"]).to_csv(history_path, index=False)
    summary = {
        "experiment": run_config.experiment_name,
        "dataset": run_config.dataset_name,
        "run_mode": run_config.run_mode,
        "architecture": run_config.architecture,
        "variant_name": run_config.variant_name,
        "loss_name": run_config.loss_name,
        "dice": result["metrics"]["dice"],
        "iou": result["metrics"]["iou"],
        "pixel_accuracy": result["metrics"]["pixel_accuracy"],
        "threshold": result.get("threshold", run_config.threshold),
        "split_sizes": result["split_sizes"],
        "hyperparameters": {
            "epochs": run_config.epochs,
            "batch_size": run_config.batch_size,
            "learning_rate": run_config.learning_rate,
            "weight_decay": run_config.weight_decay,
            "early_stopping_patience": run_config.early_stopping_patience,
            "base_features": run_config.base_features,
            "bce_weight": run_config.bce_weight,
            "dice_weight": run_config.dice_weight,
            "pos_weight": run_config.pos_weight,
            "tversky_alpha": run_config.tversky_alpha,
            "tversky_beta": run_config.tversky_beta,
            "optimize_threshold": run_config.optimize_threshold,
            "threshold_search_min": run_config.threshold_search_min,
            "threshold_search_max": run_config.threshold_search_max,
            "threshold_search_step": run_config.threshold_search_step,
            "train_crop_size": run_config.train_crop_size,
            "train_crop_prob": run_config.train_crop_prob,
            "positive_crop_prob": run_config.positive_crop_prob,
            "ct_context_slices": run_config.ct_context_slices,
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    return {"model": model_path, "history": history_path, "summary": summary_path}


def existing_segmentation_artifact(run_config: SegmentationRunConfig, models_dir: Path, results_dir: Path) -> bool:
    artifact_name = f"{run_config.experiment_name}_{run_config.run_mode}"
    return (models_dir / f"{artifact_name}.pt").exists() and (
        results_dir / f"{artifact_name}_summary.json"
    ).exists()


def summarize_segmentation_results(summary_paths: Iterable[Path]) -> pd.DataFrame:
    rows = []
    for summary_path in summary_paths:
        row = json.loads(summary_path.read_text())
        row["summary_path"] = str(summary_path)
        rows.append(row)
    return pd.DataFrame(rows)
