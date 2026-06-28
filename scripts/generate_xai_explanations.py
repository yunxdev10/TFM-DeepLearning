import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
from tqdm import tqdm

from src.config import config
from src.data.ct_preprocessing import get_ct_dataframes
from src.data.datasets import CTDataset, CXRDataset
from src.data.preprocessing import get_cxr_dataframes
from src.data.segmentation import build_ct_segmentation_dataframe
from src.data.transforms import get_ct_transforms, get_cxr_transforms
from src.evaluation.explainability import (
    GradCAM,
    binary_iou,
    denormalize_image,
    load_binary_mask,
    load_classifier_model,
    resolve_gradcam_target_layer,
    saliency_inside_mask_ratio,
    saliency_peak_inside_mask,
    saliency_to_binary_mask,
)
from src.training.classification_experiment import get_device


DEFAULT_CXR_EXPERIMENT = "cxr_densenet121_weighted_ce"
DEFAULT_CT_EXPERIMENT = "ct_densenet121_baseline"


def load_json(path: Path) -> Dict:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def load_checkpoint_label_map(model_path: Path) -> Dict[str, int]:
    checkpoint = torch.load(model_path, map_location="cpu")
    label_map = checkpoint.get("label_map") if isinstance(checkpoint, dict) else None
    if not label_map:
        raise ValueError(f"Missing label_map in checkpoint: {model_path}")
    return {str(label): int(index) for label, index in label_map.items()}


def invert_label_map(label_map: Dict[str, int]) -> Dict[int, str]:
    return {index: label for label, index in label_map.items()}


def cxr_mask_path(image_path: str) -> str:
    return str(image_path).replace("/images/", "/masks/")


def build_cxr_xai_dataframe(label_map: Dict[str, int]) -> pd.DataFrame:
    _, _, test_df = get_cxr_dataframes(config.CXR_DIR, config.RANDOM_SEED)
    df = test_df.reset_index(drop=True).copy()
    df["mask_path"] = df["image_path"].map(cxr_mask_path)
    df["sample_id"] = df["image_path"].map(lambda value: Path(value).stem)
    df["mask_available"] = df["mask_path"].map(lambda value: Path(value).exists())
    df["y_true"] = df["label"].map(label_map)
    return df[df["mask_available"]].reset_index(drop=True)


def build_ct_xai_dataframe(label_map: Dict[str, int], mask_split: str) -> pd.DataFrame:
    labels_path = config.CT_DIR / "processed_2d_slices" / "labels_metadata.csv"
    class_df = pd.read_csv(labels_path)
    train_df, val_df, test_df = get_ct_dataframes(class_df, config.RANDOM_SEED)
    split_frames = {
        "train": train_df.assign(split="train"),
        "val": val_df.assign(split="val"),
        "test": test_df.assign(split="test"),
    }
    class_split_df = pd.concat(split_frames.values(), ignore_index=True)

    ct_seg_df = build_ct_segmentation_dataframe(
        config.CT_DIR,
        config.CT_DIR / "processed_segmentation_slices",
        target_size=config.CT_IMAGE_SIZE,
        positive_mask_only=True,
        overwrite=False,
    )
    join_cols = ["study_id", "slice_index"]
    label_cols = join_cols + ["label", "split"]
    df = ct_seg_df.merge(class_split_df[label_cols], on=join_cols, how="inner")
    if mask_split != "all":
        df = df[df["split"] == mask_split]
    df = df.reset_index(drop=True)
    df["mask_available"] = df["mask_path"].map(lambda value: Path(value).exists())
    df["y_true"] = df["label_y"].map(label_map)
    df["label"] = df["label_y"]
    return df[df["mask_available"]].reset_index(drop=True)


def load_prediction_frame(results_dir: Path, experiment: str, run_mode: str, df: pd.DataFrame) -> Optional[pd.DataFrame]:
    predictions_path = results_dir / f"{experiment}_{run_mode}_predictions.csv"
    if not predictions_path.exists():
        return None

    predictions = pd.read_csv(predictions_path)
    if len(predictions) != len(df):
        return None

    merged = df.reset_index(drop=True).copy()
    merged["y_true"] = predictions["y_true"].astype(int).to_numpy()
    merged["y_pred"] = predictions["y_pred"].astype(int).to_numpy()
    prob_cols = [col for col in predictions.columns if col.startswith("prob_")]
    merged["confidence"] = predictions[prob_cols].max(axis=1).to_numpy()
    return merged


def predict_frame(
    model: torch.nn.Module,
    dataset_cls,
    df: pd.DataFrame,
    transform,
    label_map: Dict[str, int],
    device: str,
    batch_size: int,
) -> pd.DataFrame:
    dataset = dataset_cls(df, transform=transform, label_map=label_map)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    y_true: List[int] = []
    y_pred: List[int] = []
    confidence: List[float] = []
    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            logits = model(images.to(device))
            probs = torch.softmax(logits, dim=1).cpu()
            y_true.extend(labels.numpy().tolist())
            y_pred.extend(probs.argmax(dim=1).numpy().tolist())
            confidence.extend(probs.max(dim=1).values.numpy().tolist())

    output = df.reset_index(drop=True).copy()
    output["y_true"] = y_true
    output["y_pred"] = y_pred
    output["confidence"] = confidence
    return output


def select_examples(df: pd.DataFrame, max_per_class: int, max_incorrect_per_class: int) -> pd.DataFrame:
    selected = []
    for class_idx in sorted(df["y_true"].dropna().astype(int).unique().tolist()):
        class_df = df[df["y_true"] == class_idx].copy()
        correct = class_df[class_df["y_pred"] == class_idx].sort_values("confidence", ascending=False).head(max_per_class)
        incorrect = (
            class_df[class_df["y_pred"] != class_idx]
            .sort_values("confidence", ascending=False)
            .head(max_incorrect_per_class)
        )
        selected.append(correct)
        selected.append(incorrect)
    if not selected:
        return df.head(0).copy()
    return pd.concat(selected, ignore_index=False).drop_duplicates().reset_index(drop=True)


def plot_xai_example(
    output_path: Path,
    image: np.ndarray,
    mask: np.ndarray,
    heatmap: np.ndarray,
    saliency_mask: np.ndarray,
    title: str,
) -> None:
    fig, axes = plt.subplots(1, 5, figsize=(17, 3.7), dpi=160)
    if image.ndim == 2:
        axes[0].imshow(image, cmap="gray", vmin=0, vmax=1)
    else:
        axes[0].imshow(image, vmin=0, vmax=1)
    axes[0].set_title("Imagen")
    axes[1].imshow(mask.astype(float), cmap="Greens", vmin=0, vmax=1)
    axes[1].set_title("Mascara")

    base = image if image.ndim == 3 else np.repeat(image[..., None], 3, axis=-1)
    axes[2].imshow(base, vmin=0, vmax=1)
    axes[2].imshow(heatmap, cmap="jet", alpha=0.45, vmin=0, vmax=1)
    axes[2].set_title("Grad-CAM")

    axes[3].imshow(saliency_mask.astype(float), cmap="Reds", vmin=0, vmax=1)
    axes[3].set_title("Saliencia binaria")

    axes[4].imshow(base, vmin=0, vmax=1)
    axes[4].imshow(mask.astype(float), cmap="Greens", alpha=0.35, vmin=0, vmax=1)
    axes[4].imshow(saliency_mask.astype(float), cmap="Reds", alpha=0.35, vmin=0, vmax=1)
    axes[4].set_title("Mascara vs saliencia")

    for axis in axes:
        axis.axis("off")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def generate_dataset_xai(
    dataset_name: str,
    experiment: str,
    run_mode: str,
    max_per_class: int,
    max_incorrect_per_class: int,
    saliency_quantile: float,
    ct_mask_split: str,
    device: str,
) -> Dict[str, Path]:
    results_dir = PROJECT_ROOT / "results" / "classification" / dataset_name
    models_dir = config.MODELS_DIR / dataset_name
    output_name = f"{experiment}_{run_mode}"
    if dataset_name == "ct":
        output_name = f"{output_name}_{ct_mask_split}"
    output_dir = PROJECT_ROOT / "results" / "explainability" / dataset_name / output_name
    figures_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    if figures_dir.exists():
        for stale_figure in figures_dir.glob("*_gradcam.png"):
            stale_figure.unlink()

    summary = load_json(results_dir / f"{experiment}_{run_mode}_summary.json")
    model_path = models_dir / f"{experiment}_{run_mode}.pt"
    label_map = load_checkpoint_label_map(model_path)
    index_to_label = invert_label_map(label_map)
    in_channels = 3 if dataset_name == "cxr" else 1
    image_size = config.CXR_IMAGE_SIZE if dataset_name == "cxr" else config.CT_IMAGE_SIZE
    mean = (0.485, 0.456, 0.406) if dataset_name == "cxr" else (0.5,)
    std = (0.229, 0.224, 0.225) if dataset_name == "cxr" else (0.5,)
    transform = get_cxr_transforms(config.CXR_IMAGE_SIZE, is_train=False) if dataset_name == "cxr" else get_ct_transforms(config.CT_IMAGE_SIZE, is_train=False)
    dataset_cls = CXRDataset if dataset_name == "cxr" else CTDataset

    model, _ = load_classifier_model(
        model_path=model_path,
        architecture=summary["architecture"],
        num_classes=len(label_map),
        in_channels=in_channels,
        device=device,
    )
    xai_df = build_cxr_xai_dataframe(label_map) if dataset_name == "cxr" else build_ct_xai_dataframe(label_map, ct_mask_split)
    prediction_df = load_prediction_frame(results_dir, experiment, run_mode, xai_df)
    if prediction_df is None:
        prediction_df = predict_frame(
            model=model,
            dataset_cls=dataset_cls,
            df=xai_df,
            transform=transform,
            label_map=label_map,
            device=device,
            batch_size=16,
        )

    selected_df = select_examples(prediction_df, max_per_class, max_incorrect_per_class)
    if selected_df.empty:
        raise ValueError(f"No examples selected for dataset={dataset_name}, experiment={experiment}.")

    dataset = dataset_cls(selected_df, transform=transform, label_map=label_map)
    grad_cam = GradCAM(model, resolve_gradcam_target_layer(model))
    records = []
    try:
        for row_idx, (image_tensor, label_tensor) in enumerate(tqdm(dataset, desc=f"{dataset_name} Grad-CAM")):
            row = selected_df.iloc[row_idx]
            input_tensor = image_tensor.unsqueeze(0).to(device)
            target_class = int(row["y_pred"])
            cam_tensor, logits = grad_cam(input_tensor, target_class=target_class)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            heatmap = cam_tensor[0, 0].cpu().numpy()
            mask = load_binary_mask(Path(row["mask_path"]), image_size=image_size)
            saliency_mask = saliency_to_binary_mask(heatmap, quantile=saliency_quantile)
            image = denormalize_image(image_tensor, mean=mean, std=std)

            sample_id = row.get("sample_id", Path(row["image_path"]).stem)
            true_label = index_to_label[int(row["y_true"])]
            pred_label = index_to_label[int(row["y_pred"])]
            iou = binary_iou(saliency_mask, mask)
            inside_ratio = saliency_inside_mask_ratio(heatmap, mask)
            peak_inside = saliency_peak_inside_mask(heatmap, mask)
            figure_path = figures_dir / f"{row_idx:03d}_{sample_id}_gradcam.png"

            plot_xai_example(
                output_path=figure_path,
                image=image,
                mask=mask,
                heatmap=heatmap,
                saliency_mask=saliency_mask,
                title=(
                    f"{dataset_name.upper()} | {sample_id} | true={true_label} pred={pred_label} "
                    f"| conf={float(probs[target_class]):.3f} | IoU={iou:.3f}"
                ),
            )
            records.append(
                {
                    "dataset": dataset_name,
                    "experiment": experiment,
                    "method": "grad_cam",
                    "sample_id": sample_id,
                    "image_path": row["image_path"],
                    "mask_path": row["mask_path"],
                    "true_label": true_label,
                    "pred_label": pred_label,
                    "is_correct": bool(int(row["y_true"]) == int(row["y_pred"])),
                    "confidence": float(probs[target_class]),
                    "saliency_mask_iou": iou,
                    "saliency_inside_mask_ratio": inside_ratio,
                    "saliency_peak_inside_mask": peak_inside,
                    "figure_path": str(figure_path),
                }
            )
    finally:
        grad_cam.close()

    metrics_df = pd.DataFrame(records)
    metrics_path = output_dir / f"{experiment}_{run_mode}_xai_metrics.csv"
    aggregate_path = output_dir / f"{experiment}_{run_mode}_xai_summary.json"
    metrics_df.to_csv(metrics_path, index=False)
    aggregate = {
        "dataset": dataset_name,
        "experiment": experiment,
        "run_mode": run_mode,
        "architecture": summary["architecture"],
        "mask_reference": "lung_mask" if dataset_name == "cxr" else "ct_infection_mask",
        "method": "grad_cam",
        "num_examples": int(len(metrics_df)),
        "saliency_quantile": saliency_quantile,
        "mean_saliency_mask_iou": float(metrics_df["saliency_mask_iou"].mean()),
        "mean_saliency_inside_mask_ratio": float(metrics_df["saliency_inside_mask_ratio"].mean()),
        "peak_inside_mask_rate": float(metrics_df["saliency_peak_inside_mask"].mean()),
        "note": (
            "CXR masks are lung-field masks, so IoU measures anatomical plausibility, not COVID lesion localization."
            if dataset_name == "cxr"
            else "CT masks are infection masks available only for annotated MosMedData studies/slices."
        ),
    }
    aggregate_path.write_text(json.dumps(aggregate, indent=2))
    print(f"\n=== {dataset_name.upper()} {experiment} ===")
    print(metrics_df[["sample_id", "true_label", "pred_label", "is_correct", "saliency_mask_iou", "saliency_inside_mask_ratio", "saliency_peak_inside_mask"]])
    print(f"Saved metrics: {metrics_path}")
    print(f"Saved summary: {aggregate_path}")
    return {"metrics": metrics_path, "summary": aggregate_path, "output_dir": output_dir}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Grad-CAM XAI examples and saliency-mask metrics.")
    parser.add_argument("--dataset", choices=["cxr", "ct", "both"], default="both")
    parser.add_argument("--cxr-experiment", default=DEFAULT_CXR_EXPERIMENT)
    parser.add_argument("--ct-experiment", default=DEFAULT_CT_EXPERIMENT)
    parser.add_argument("--run-mode", default="full")
    parser.add_argument("--max-per-class", type=int, default=2)
    parser.add_argument("--max-incorrect-per-class", type=int, default=1)
    parser.add_argument("--saliency-quantile", type=float, default=0.80)
    parser.add_argument("--ct-mask-split", choices=["train", "val", "test", "all"], default="test")
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config.NUM_WORKERS = 0
    (PROJECT_ROOT / "results" / ".matplotlib").mkdir(parents=True, exist_ok=True)
    device = get_device() if args.device == "auto" else args.device
    datasets = ["cxr", "ct"] if args.dataset == "both" else [args.dataset]
    outputs = []
    for dataset_name in datasets:
        experiment = args.cxr_experiment if dataset_name == "cxr" else args.ct_experiment
        outputs.append(
            generate_dataset_xai(
                dataset_name=dataset_name,
                experiment=experiment,
                run_mode=args.run_mode,
                max_per_class=args.max_per_class,
                max_incorrect_per_class=args.max_incorrect_per_class,
                saliency_quantile=args.saliency_quantile,
                ct_mask_split=args.ct_mask_split,
                device=device,
            )
        )

    print("\nGenerated XAI outputs:")
    for output in outputs:
        print(output["output_dir"])


if __name__ == "__main__":
    main()
