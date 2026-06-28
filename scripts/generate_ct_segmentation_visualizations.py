import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / "results" / ".matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.config import config
from src.data.segmentation import (
    SegmentationDataset,
    SegmentationPairTransform,
    build_ct_segmentation_dataframe,
    split_segmentation_dataframe,
)
from src.models.segmentation import build_segmentation_model
from src.training.segmentation_experiment import get_device


DEFAULT_EXPERIMENT = "ct_attention_unet_mixed30_patch192_pos70_tversky_pos10_bf32_thr095_segmentation"


def load_summary(results_dir: Path, experiment: str, run_mode: str) -> Dict:
    summary_path = results_dir / f"{experiment}_{run_mode}_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary: {summary_path}")
    return json.loads(summary_path.read_text())


def load_checkpoint_state_dict(model_path: Path) -> Dict:
    checkpoint = torch.load(model_path, map_location="cpu")
    return checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint


def infer_checkpoint_in_channels(model_path: Path) -> Optional[int]:
    if not model_path.exists():
        return None

    state_dict = load_checkpoint_state_dict(model_path)
    first_weight = state_dict.get("down_blocks.0.block.0.weight")
    if first_weight is not None and first_weight.ndim >= 2:
        return int(first_weight.shape[1])
    return None


def infer_model_in_channels(summary: Dict, model_path: Path) -> int:
    hyperparameters = summary.get("hyperparameters", {})
    explicit_in_channels = summary.get("in_channels", hyperparameters.get("in_channels"))
    if explicit_in_channels is not None:
        return int(explicit_in_channels)

    if bool(hyperparameters.get("ct_context_slices", False)):
        return 3

    checkpoint_in_channels = infer_checkpoint_in_channels(model_path)
    if checkpoint_in_channels is not None:
        return checkpoint_in_channels

    return 1


def load_model(model_path: Path, architecture: str, in_channels: int, base_features: int, device: str) -> torch.nn.Module:
    model = build_segmentation_model(
        architecture=architecture,
        in_channels=in_channels,
        out_channels=1,
        base_features=base_features,
    )
    state_dict = load_checkpoint_state_dict(model_path)
    model.load_state_dict(state_dict)
    return model.to(device).eval()


def collect_display_tensors(df: pd.DataFrame, image_size) -> Dict[str, torch.Tensor]:
    transform = SegmentationPairTransform(image_size=image_size, in_channels=1, is_train=False)
    dataset = SegmentationDataset(df, transform=transform, in_channels=1, ct_context_slices=False)
    loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)

    images = []
    masks = []
    with torch.no_grad():
        for batch_images, batch_masks in loader:
            images.append(batch_images.cpu())
            masks.append(batch_masks.cpu())

    return {"images": torch.cat(images), "masks": torch.cat(masks)}


def collect_model_probabilities(
    model: torch.nn.Module,
    df: pd.DataFrame,
    image_size,
    in_channels: int,
    ct_context_slices: bool,
    device: str,
) -> torch.Tensor:
    transform = SegmentationPairTransform(image_size=image_size, in_channels=in_channels, is_train=False)
    dataset = SegmentationDataset(
        df,
        transform=transform,
        in_channels=in_channels,
        ct_context_slices=ct_context_slices,
    )
    loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)

    probs = []
    with torch.no_grad():
        for batch_images, _ in loader:
            logits = model(batch_images.to(device))
            probs.append(torch.sigmoid(logits).cpu())

    return torch.cat(probs)


def discover_ct_summaries(results_dir: Path, run_mode: str, include_ensembles: bool = True) -> List[Dict]:
    summaries = []
    for summary_path in sorted(results_dir.glob(f"*_segmentation_{run_mode}_summary.json")):
        summary = json.loads(summary_path.read_text())
        if summary.get("dataset") != "ct" or summary.get("run_mode") != run_mode:
            continue
        if not include_ensembles and summary.get("loss_name") == "probability_average":
            continue
        summary["_summary_path"] = str(summary_path)
        summaries.append(summary)
    return summaries


def get_individual_model_spec(summary: Dict, results_dir: Path, models_dir: Path, run_mode: str) -> Dict:
    experiment = summary["experiment"]
    hyperparameters = summary.get("hyperparameters", {})
    model_path = models_dir / f"{experiment}_{run_mode}.pt"
    return {
        "experiment": experiment,
        "architecture": summary.get("architecture", "attention_unet"),
        "in_channels": infer_model_in_channels(summary, model_path),
        "base_features": int(hyperparameters.get("base_features", 32)),
        "ct_context_slices": bool(hyperparameters.get("ct_context_slices", False)),
        "model_path": model_path,
        "weight": 1.0,
    }


def get_ensemble_model_specs(summary: Dict, results_dir: Path, models_dir: Path, run_mode: str) -> List[Dict]:
    hyperparameters = summary.get("hyperparameters", {})
    ensemble_members = [
        (hyperparameters.get("old_model"), float(hyperparameters.get("old_weight", 0.0))),
        (hyperparameters.get("new_model"), float(hyperparameters.get("new_weight", 0.0))),
    ]
    specs = []
    for experiment, weight in ensemble_members:
        if not experiment or weight <= 0:
            continue
        member_summary = load_summary(results_dir, experiment, run_mode)
        spec = get_individual_model_spec(member_summary, results_dir, models_dir, run_mode)
        spec["weight"] = weight
        specs.append(spec)
    if not specs:
        raise ValueError(f"No valid ensemble members found for {summary['experiment']}")
    return specs


def get_model_specs(summary: Dict, results_dir: Path, models_dir: Path, run_mode: str) -> List[Dict]:
    if summary.get("loss_name") == "probability_average":
        return get_ensemble_model_specs(summary, results_dir, models_dir, run_mode)
    return [get_individual_model_spec(summary, results_dir, models_dir, run_mode)]


def collect_weighted_probabilities(
    model_specs: Iterable[Dict],
    df: pd.DataFrame,
    image_size,
    device: str,
) -> torch.Tensor:
    weighted_probs = None
    total_weight = 0.0
    for spec in model_specs:
        model = load_model(
            spec["model_path"],
            spec["architecture"],
            spec["in_channels"],
            spec["base_features"],
            device,
        )
        probs = collect_model_probabilities(
            model=model,
            df=df,
            image_size=image_size,
            in_channels=spec["in_channels"],
            ct_context_slices=spec["ct_context_slices"],
            device=device,
        )
        weight = float(spec.get("weight", 1.0))
        weighted_probs = probs * weight if weighted_probs is None else weighted_probs + probs * weight
        total_weight += weight
    if weighted_probs is None or total_weight <= 0:
        raise ValueError("No probabilities were collected.")
    return weighted_probs / total_weight


def compute_slice_metrics(df: pd.DataFrame, masks: torch.Tensor, preds: torch.Tensor) -> pd.DataFrame:
    dims = (1, 2, 3)
    intersection = (preds * masks).sum(dims).numpy()
    pred_sum = preds.sum(dims).numpy()
    target_sum = masks.sum(dims).numpy()
    union = pred_sum + target_sum - intersection
    dice = (2 * intersection + 1) / (pred_sum + target_sum + 1)
    iou = (intersection + 1) / (union + 1)

    metrics = df[["sample_id", "study_id", "slice_index"]].copy()
    metrics["target_px"] = target_sum
    metrics["pred_px"] = pred_sum
    metrics["intersection_px"] = intersection
    metrics["dice"] = dice
    metrics["iou"] = iou
    metrics["false_positive_px"] = pred_sum - intersection
    metrics["false_negative_px"] = target_sum - intersection
    return metrics


def select_representative_indices(metrics: pd.DataFrame, rows_per_group: int = 3) -> List[int]:
    worst = metrics.sort_values("dice", ascending=True).head(rows_per_group).index.tolist()
    best = metrics.sort_values("dice", ascending=False).head(rows_per_group).index.tolist()
    median_value = float(metrics["dice"].median())
    median = (metrics.assign(distance=(metrics["dice"] - median_value).abs()).sort_values("distance").head(rows_per_group).index.tolist())

    selected = []
    for idx in worst + median + best:
        if idx not in selected:
            selected.append(idx)
    return selected


def make_overlay(image: np.ndarray, mask: np.ndarray, pred: np.ndarray) -> np.ndarray:
    base = np.repeat(image[..., None], 3, axis=-1)
    base = np.clip(base, 0, 1)
    overlay = base.copy()
    overlay[mask] = overlay[mask] * 0.45 + np.array([0.0, 0.9, 0.1]) * 0.55
    overlay[pred] = overlay[pred] * 0.45 + np.array([1.0, 0.05, 0.0]) * 0.55
    return overlay


def make_error_map(image: np.ndarray, mask: np.ndarray, pred: np.ndarray) -> np.ndarray:
    error_map = np.repeat((image * 0.25)[..., None], 3, axis=-1)
    true_positive = mask & pred
    false_positive = ~mask & pred
    false_negative = mask & ~pred
    error_map[true_positive] = np.array([0.0, 0.8, 0.1])
    error_map[false_positive] = np.array([1.0, 0.05, 0.0])
    error_map[false_negative] = np.array([0.1, 0.35, 1.0])
    return error_map


def plot_examples(
    output_path: Path,
    selected_indices: List[int],
    metrics: pd.DataFrame,
    tensors: Dict[str, torch.Tensor],
    experiment_label: str,
    threshold: float,
) -> None:
    num_rows = len(selected_indices)
    fig, axes = plt.subplots(num_rows, 5, figsize=(17, max(3.0, 2.9 * num_rows)), dpi=160)
    if num_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    column_titles = ["CT", "Mascara real", "Prediccion", "Overlay", "Errores"]
    for col, title in enumerate(column_titles):
        axes[0, col].set_title(title, fontsize=11)

    for row_idx, sample_idx in enumerate(selected_indices):
        image = tensors["images"][sample_idx, 0].numpy()
        mask = tensors["masks"][sample_idx, 0].numpy().astype(bool)
        pred = tensors["preds"][sample_idx, 0].numpy().astype(bool)
        metric_row = metrics.loc[sample_idx]

        panels = [
            image,
            mask.astype(float),
            pred.astype(float),
            make_overlay(image, mask, pred),
            make_error_map(image, mask, pred),
        ]
        cmaps = ["gray", "Greens", "Reds", None, None]
        for col, panel in enumerate(panels):
            axes[row_idx, col].imshow(panel, cmap=cmaps[col], vmin=0, vmax=1)
            axes[row_idx, col].axis("off")

        axes[row_idx, 0].text(
            0.02,
            0.98,
            f"{metric_row.sample_id}\nDice={metric_row.dice:.3f} IoU={metric_row.iou:.3f}\n"
            f"GT={int(metric_row.target_px)} Pred={int(metric_row.pred_px)}",
            transform=axes[row_idx, 0].transAxes,
            fontsize=7,
            color="white",
            va="top",
            ha="left",
            bbox={"facecolor": "black", "alpha": 0.55, "pad": 2, "edgecolor": "none"},
        )

    fig.suptitle(
        f"CT segmentation qualitative examples | {experiment_label} | threshold={threshold:.2f}\n"
        "Overlay: verde=mascara real, rojo=prediccion. Errores: verde=TP, rojo=FP, azul=FN.",
        fontsize=12,
        y=0.995,
    )
    fig.tight_layout(rect=(0.02, 0.0, 1.0, 0.975))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate qualitative CT segmentation visualizations.")
    parser.add_argument("--experiment", default=DEFAULT_EXPERIMENT)
    parser.add_argument("--all-ct", action="store_true", help="Generate one separate folder/figure for each CT experiment summary.")
    parser.add_argument("--list-experiments", action="store_true", help="List CT experiments detected and exit.")
    parser.add_argument("--exclude-ensembles", action="store_true", help="Skip probability-average ensemble summaries in --all-ct mode.")
    parser.add_argument(
        "--shared-selection-experiment",
        default=DEFAULT_EXPERIMENT,
        help="In --all-ct mode, reuse representative test slices selected from this experiment. Use 'none' for per-model selection.",
    )
    parser.add_argument("--run-mode", default="full")
    parser.add_argument("--rows-per-group", type=int, default=3)
    parser.add_argument("--output-dir", default="results/segmentation/ct/qualitative")
    return parser.parse_args()


def generate_for_summary(
    summary: Dict,
    test_df: pd.DataFrame,
    display_tensors: Dict[str, torch.Tensor],
    results_dir: Path,
    models_dir: Path,
    output_dir: Path,
    run_mode: str,
    rows_per_group: int,
    device: str,
    selected_indices: Optional[List[int]] = None,
) -> Dict[str, Path]:
    hyperparameters = summary.get("hyperparameters", {})
    threshold = float(summary.get("threshold", 0.5))
    experiment = summary["experiment"]

    model_specs = get_model_specs(summary, results_dir, models_dir, run_mode)
    probs = collect_weighted_probabilities(model_specs, test_df, config.CT_IMAGE_SIZE, device)
    preds = (probs >= threshold).float()
    tensors = {
        "images": display_tensors["images"],
        "masks": display_tensors["masks"],
        "probs": probs,
        "preds": preds,
    }

    metrics = compute_slice_metrics(test_df, tensors["masks"], tensors["preds"])
    if selected_indices is None:
        selected_indices = select_representative_indices(metrics, rows_per_group=rows_per_group)

    metrics_path = output_dir / f"{experiment}_{run_mode}_test_slice_metrics.csv"
    selected_path = output_dir / f"{experiment}_{run_mode}_selected_examples.csv"
    figure_path = output_dir / f"{experiment}_{run_mode}_qualitative_grid.png"

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(metrics_path, index=False)
    metrics.loc[selected_indices].to_csv(selected_path, index=False)
    plot_examples(
        output_path=figure_path,
        selected_indices=selected_indices,
        metrics=metrics,
        tensors=tensors,
        experiment_label=experiment,
        threshold=threshold,
    )

    print(f"Saved metrics: {metrics_path}")
    print(f"Saved selected examples: {selected_path}")
    print(f"Saved figure: {figure_path}")
    print(metrics.loc[selected_indices, ["sample_id", "study_id", "slice_index", "target_px", "pred_px", "dice", "iou"]])
    return {"metrics": metrics_path, "selected": selected_path, "figure": figure_path}


def main() -> None:
    args = parse_args()
    config.NUM_WORKERS = 0
    (PROJECT_ROOT / "results" / ".matplotlib").mkdir(parents=True, exist_ok=True)
    results_dir = PROJECT_ROOT / "results" / "segmentation" / "ct"
    models_dir = config.MODELS_DIR / "segmentation" / "ct"
    output_dir = PROJECT_ROOT / args.output_dir

    summaries = discover_ct_summaries(
        results_dir=results_dir,
        run_mode=args.run_mode,
        include_ensembles=not args.exclude_ensembles,
    )
    if args.list_experiments:
        for summary in sorted(summaries, key=lambda item: item.get("dice", -1), reverse=True):
            print(
                f"{summary['experiment']} | dice={summary.get('dice', float('nan')):.4f} "
                f"| iou={summary.get('iou', float('nan')):.4f} | threshold={summary.get('threshold', 'NA')}"
            )
        return

    if args.all_ct:
        selected_summaries = summaries
    else:
        selected_summaries = [load_summary(results_dir, args.experiment, args.run_mode)]

    ct_df = build_ct_segmentation_dataframe(
        config.CT_DIR,
        config.CT_DIR / "processed_segmentation_slices",
        target_size=config.CT_IMAGE_SIZE,
        positive_mask_only=True,
        overwrite=False,
    )
    _, _, test_df = split_segmentation_dataframe(ct_df, random_seed=config.RANDOM_SEED, group_col="study_id")
    display_tensors = collect_display_tensors(test_df, config.CT_IMAGE_SIZE)
    device = get_device()

    shared_indices = None
    if args.all_ct and args.shared_selection_experiment.lower() != "none":
        shared_summary = load_summary(results_dir, args.shared_selection_experiment, args.run_mode)
        shared_specs = get_model_specs(shared_summary, results_dir, models_dir, args.run_mode)
        shared_probs = collect_weighted_probabilities(shared_specs, test_df, config.CT_IMAGE_SIZE, device)
        shared_preds = (shared_probs >= float(shared_summary.get("threshold", 0.5))).float()
        shared_metrics = compute_slice_metrics(test_df, display_tensors["masks"], shared_preds)
        shared_indices = select_representative_indices(shared_metrics, rows_per_group=args.rows_per_group)
        print(f"Using shared selected slices from {args.shared_selection_experiment}: {shared_indices}")

    generated_paths = []
    for summary in sorted(selected_summaries, key=lambda item: item.get("dice", -1), reverse=True):
        experiment_output_dir = output_dir / summary["experiment"] if args.all_ct else output_dir
        print(f"\n=== {summary['experiment']} ===")
        generated_paths.append(
            generate_for_summary(
                summary=summary,
                test_df=test_df,
                display_tensors=display_tensors,
                results_dir=results_dir,
                models_dir=models_dir,
                output_dir=experiment_output_dir,
                run_mode=args.run_mode,
                rows_per_group=args.rows_per_group,
                device=device,
                selected_indices=shared_indices,
            )
        )

    print("\nGenerated figures:")
    for paths in generated_paths:
        print(paths["figure"])


if __name__ == "__main__":
    main()
