#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / "results" / ".matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import config
from src.data.ct_preprocessing import get_ct_dataframes
from src.data.datasets import CTDataset
from src.data.transforms import get_ct_transforms
from src.models.classifiers import CovidClassifier
from src.training.classification_experiment import get_device, seed_everything


LABEL_MAP = {"CT-0": 0, "CT-1": 1, "CT-2": 2, "CT-3+": 3}
CLASS_NAMES = {value: key for key, value in LABEL_MAP.items()}
PROB_COLUMNS = [f"prob_{idx}" for idx in range(len(LABEL_MAP))]
RESULTS_DIR = PROJECT_ROOT / "results" / "classification" / "ct_study_level_meta"
FIGURES_DIR = RESULTS_DIR / "figures"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a lightweight CT study-level meta-classifier from informative-slice probabilities.",
    )
    parser.add_argument(
        "--base-experiment",
        default="ct_top20_tissue_resnet50_weighted_ce",
        help="Informative-slice experiment used as feature extractor.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size used for inference with the trained slice model.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader workers for deterministic local execution.",
    )
    parser.add_argument(
        "--force-inference",
        action="store_true",
        help="Regenerate train/val/test slice probabilities even if cached CSV files exist.",
    )
    return parser.parse_args()


def softmax_entropy(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(probabilities, 1e-8, 1.0)
    return -(clipped * np.log(clipped)).sum(axis=1)


def load_summary(base_experiment: str) -> dict:
    summary_path = PROJECT_ROOT / "results" / "classification" / "ct_informative_slices" / f"{base_experiment}_full_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(
            f"Missing summary for {base_experiment}. Expected: {summary_path}"
        )
    return json.loads(summary_path.read_text())


def split_metadata(summary: dict) -> dict[str, pd.DataFrame]:
    metadata_path = Path(summary["metadata_path"])
    metadata_df = pd.read_csv(metadata_path)
    train_df, val_df, test_df = get_ct_dataframes(metadata_df, config.RANDOM_SEED)
    return {
        "train": train_df.reset_index(drop=True),
        "val": val_df.reset_index(drop=True),
        "test": test_df.reset_index(drop=True),
    }


def load_slice_model(summary: dict, device: str) -> CovidClassifier:
    model_path = PROJECT_ROOT / "models" / "ct_informative_slices" / f"{summary['experiment']}_full.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing trained slice model: {model_path}")

    checkpoint = torch.load(model_path, map_location="cpu")
    architecture = checkpoint.get("architecture", summary["architecture"])
    model = CovidClassifier(
        architecture_name=architecture,
        num_classes=len(LABEL_MAP),
        in_channels=1,
        pretrained=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def run_inference_for_split(
    split_name: str,
    split_df: pd.DataFrame,
    model: CovidClassifier,
    device: str,
    batch_size: int,
    num_workers: int,
    output_path: Path,
    force: bool,
) -> pd.DataFrame:
    if output_path.exists() and not force:
        print(f"Using cached {split_name} slice probabilities: {output_path}", flush=True)
        return pd.read_csv(output_path)

    print(f"Running slice inference for {split_name}: {len(split_df)} slices", flush=True)
    dataset = CTDataset(split_df, transform=get_ct_transforms(config.CT_IMAGE_SIZE, is_train=False), label_map=LABEL_MAP)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    y_prob: list[np.ndarray] = []
    y_true: list[np.ndarray] = []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            probabilities = torch.softmax(model(inputs), dim=1).cpu().numpy()
            y_prob.append(probabilities)
            y_true.append(labels.numpy())

    probabilities = np.concatenate(y_prob, axis=0)
    true_labels = np.concatenate(y_true, axis=0).astype(int)
    pred_labels = probabilities.argmax(axis=1).astype(int)

    enriched = split_df[
        ["study_id", "image_path", "slice_index", "total_slices", "label"]
    ].reset_index(drop=True)
    enriched.insert(0, "split", split_name)
    enriched["y_true"] = true_labels
    enriched["y_pred"] = pred_labels
    for idx, column in enumerate(PROB_COLUMNS):
        enriched[column] = probabilities[:, idx]
    enriched["slice_confidence"] = probabilities.max(axis=1)
    enriched["slice_entropy"] = softmax_entropy(probabilities)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_csv(output_path, index=False)
    print(f"Saved {split_name} slice probabilities: {output_path}", flush=True)
    return enriched


def collect_slice_predictions(
    summary: dict,
    splits: dict[str, pd.DataFrame],
    device: str,
    batch_size: int,
    num_workers: int,
    force: bool,
) -> pd.DataFrame:
    model = load_slice_model(summary, device)
    frames = []
    for split_name, split_df in splits.items():
        path = RESULTS_DIR / f"{summary['experiment']}_full_{split_name}_slice_predictions.csv"
        frames.append(
            run_inference_for_split(
                split_name=split_name,
                split_df=split_df,
                model=model,
                device=device,
                batch_size=batch_size,
                num_workers=num_workers,
                output_path=path,
                force=force,
            )
        )
    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(RESULTS_DIR / f"{summary['experiment']}_full_all_splits_slice_predictions.csv", index=False)
    return combined


def normalize(scores: np.ndarray) -> np.ndarray:
    scores = np.clip(np.asarray(scores, dtype=float), 0.0, None)
    total = scores.sum()
    if total <= 0:
        return np.ones_like(scores) / len(scores)
    return scores / total


def aggregate_rule(group: pd.DataFrame, aggregation: str) -> np.ndarray:
    probabilities = group[PROB_COLUMNS].to_numpy(dtype=float)
    if aggregation == "mean_probability":
        return normalize(probabilities.mean(axis=0))
    if aggregation == "confidence_weighted_mean":
        weights = group["slice_confidence"].to_numpy(dtype=float)
        weights = weights / weights.sum() if weights.sum() > 0 else np.ones(len(group)) / len(group)
        return normalize((probabilities * weights[:, None]).sum(axis=0))
    if aggregation == "majority_vote":
        votes = np.bincount(group["y_pred"].to_numpy(dtype=int), minlength=len(LABEL_MAP)).astype(float)
        return normalize(votes)
    raise ValueError(f"Unknown aggregation: {aggregation}")


def make_study_feature_rows(slice_predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (split_name, study_id), group in slice_predictions.groupby(["split", "study_id"], sort=True):
        probabilities = group[PROB_COLUMNS].to_numpy(dtype=float)
        row = {
            "split": split_name,
            "study_id": study_id,
            "label": group["label"].iloc[0],
            "y_true": int(group["y_true"].iloc[0]),
            "n_slices": int(len(group)),
            "mean_slice_confidence": float(group["slice_confidence"].mean()),
            "max_slice_confidence": float(group["slice_confidence"].max()),
            "std_slice_confidence": float(group["slice_confidence"].std(ddof=0)),
            "mean_slice_entropy": float(group["slice_entropy"].mean()),
            "min_slice_entropy": float(group["slice_entropy"].min()),
            "mean_relative_position": float((group["slice_index"] / group["total_slices"]).mean()),
            "std_relative_position": float((group["slice_index"] / group["total_slices"]).std(ddof=0)),
        }
        for class_idx in range(len(LABEL_MAP)):
            class_probs = probabilities[:, class_idx]
            sorted_probs = np.sort(class_probs)
            row[f"prob_{class_idx}_mean"] = float(class_probs.mean())
            row[f"prob_{class_idx}_max"] = float(class_probs.max())
            row[f"prob_{class_idx}_std"] = float(class_probs.std(ddof=0))
            row[f"prob_{class_idx}_median"] = float(np.median(class_probs))
            row[f"prob_{class_idx}_q75"] = float(np.quantile(class_probs, 0.75))
            row[f"prob_{class_idx}_q90"] = float(np.quantile(class_probs, 0.90))
            row[f"prob_{class_idx}_top3_mean"] = float(sorted_probs[-min(3, len(sorted_probs)):].mean())
            row[f"vote_frac_{class_idx}"] = float((group["y_pred"].to_numpy(dtype=int) == class_idx).mean())

        for aggregation in ("mean_probability", "confidence_weighted_mean", "majority_vote"):
            scores = aggregate_rule(group, aggregation)
            for class_idx in range(len(LABEL_MAP)):
                row[f"{aggregation}_prob_{class_idx}"] = float(scores[class_idx])
            row[f"{aggregation}_pred"] = int(np.argmax(scores))
        rows.append(row)
    return pd.DataFrame(rows)


def feature_columns(study_features: pd.DataFrame) -> list[str]:
    excluded = {"split", "study_id", "label", "y_true"}
    return [
        column
        for column in study_features.columns
        if column not in excluded and not column.endswith("_pred")
    ]


def proba_with_all_classes(model, x: pd.DataFrame) -> np.ndarray:
    probabilities = model.predict_proba(x)
    classes = getattr(model, "classes_", None)
    if classes is None and hasattr(model, "named_steps"):
        classes = next(reversed(model.named_steps.values())).classes_
    if classes is None:
        raise AttributeError("The fitted model does not expose classes_.")
    expanded = np.zeros((len(x), len(LABEL_MAP)), dtype=float)
    for position, class_idx in enumerate(classes):
        expanded[:, int(class_idx)] = probabilities[:, position]
    return expanded


def metric_row(
    method: str,
    split_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    source: str,
) -> dict:
    try:
        auc = float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro"))
    except ValueError:
        auc = np.nan
    return {
        "method": method,
        "split": split_name,
        "source": source,
        "n_studies": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "auc_roc_macro": auc,
    }


def prediction_frame(
    study_features: pd.DataFrame,
    method: str,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    source: str,
) -> pd.DataFrame:
    frame = study_features[["split", "study_id", "label", "y_true"]].copy()
    frame.insert(0, "method", method)
    frame.insert(1, "source", source)
    frame["y_pred"] = y_pred.astype(int)
    frame["y_true_label"] = frame["y_true"].map(CLASS_NAMES)
    frame["y_pred_label"] = frame["y_pred"].map(CLASS_NAMES)
    for class_idx in range(len(LABEL_MAP)):
        frame[f"study_prob_{class_idx}"] = y_prob[:, class_idx]
    return frame


def evaluate_rule_aggregations(study_features: pd.DataFrame) -> tuple[list[dict], list[pd.DataFrame]]:
    metrics = []
    predictions = []
    for split_name, split_df in study_features.groupby("split", sort=True):
        y_true = split_df["y_true"].to_numpy(dtype=int)
        for aggregation in ("mean_probability", "confidence_weighted_mean", "majority_vote"):
            prob_columns = [f"{aggregation}_prob_{class_idx}" for class_idx in range(len(LABEL_MAP))]
            y_prob = split_df[prob_columns].to_numpy(dtype=float)
            y_pred = y_prob.argmax(axis=1).astype(int)
            method = f"rule_{aggregation}"
            metrics.append(metric_row(method, split_name, y_true, y_pred, y_prob, "slice_probabilities"))
            predictions.append(prediction_frame(split_df, method, y_pred, y_prob, "slice_probabilities"))
    return metrics, predictions


def candidate_models() -> dict[str, object]:
    candidates: dict[str, object] = {}
    for c_value in (0.03, 0.1, 0.3, 1.0, 3.0, 10.0):
        candidates[f"logistic_balanced_c{c_value:g}"] = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=c_value,
                class_weight="balanced",
                max_iter=5000,
                random_state=config.RANDOM_SEED,
            ),
        )
    for max_depth in (2, 3, 4, None):
        name = "random_forest_balanced_depth" + ("none" if max_depth is None else str(max_depth))
        candidates[name] = RandomForestClassifier(
            n_estimators=300,
            max_depth=max_depth,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=config.RANDOM_SEED,
        )
    return candidates


def train_meta_models(study_features: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    features = feature_columns(study_features)
    train_df = study_features[study_features["split"] == "train"].reset_index(drop=True)
    val_df = study_features[study_features["split"] == "val"].reset_index(drop=True)
    test_df = study_features[study_features["split"] == "test"].reset_index(drop=True)

    x_train = train_df[features]
    y_train = train_df["y_true"].to_numpy(dtype=int)
    x_val = val_df[features]
    y_val = val_df["y_true"].to_numpy(dtype=int)
    x_test = test_df[features]
    y_test = test_df["y_true"].to_numpy(dtype=int)

    val_rows = []
    trained_models = {}
    for method, model in candidate_models().items():
        model.fit(x_train, y_train)
        val_prob = proba_with_all_classes(model, x_val)
        val_pred = val_prob.argmax(axis=1).astype(int)
        val_rows.append(metric_row(method, "val", y_val, val_pred, val_prob, "study_features"))
        trained_models[method] = model

    validation_df = pd.DataFrame(val_rows).sort_values(
        ["f1_macro", "auc_roc_macro", "accuracy"],
        ascending=False,
    )
    best_method = validation_df.iloc[0]["method"]

    final_model = candidate_models()[best_method]
    train_val_df = study_features[study_features["split"].isin(["train", "val"])].reset_index(drop=True)
    final_model.fit(train_val_df[features], train_val_df["y_true"].to_numpy(dtype=int))

    test_prob = proba_with_all_classes(final_model, x_test)
    test_pred = test_prob.argmax(axis=1).astype(int)
    test_metrics = pd.DataFrame(
        [metric_row(f"meta_{best_method}", "test", y_test, test_pred, test_prob, "study_features")]
    )
    test_predictions = prediction_frame(test_df, f"meta_{best_method}", test_pred, test_prob, "study_features")

    validation_df["selected_for_test"] = validation_df["method"].eq(best_method)
    return validation_df, test_metrics, test_predictions


def build_reports_and_confusions(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    reports = []
    confusions = []
    for (method, split_name), group in predictions.groupby(["method", "split"], sort=True):
        report = classification_report(
            group["y_true"],
            group["y_pred"],
            labels=list(CLASS_NAMES),
            target_names=[CLASS_NAMES[idx] for idx in CLASS_NAMES],
            zero_division=0,
            output_dict=True,
        )
        report_df = pd.DataFrame(report).T.reset_index(names="label")
        report_df.insert(0, "split", split_name)
        report_df.insert(0, "method", method)
        reports.append(report_df)

        matrix = confusion_matrix(group["y_true"], group["y_pred"], labels=list(CLASS_NAMES))
        for true_idx, true_label in CLASS_NAMES.items():
            for pred_idx, pred_label in CLASS_NAMES.items():
                confusions.append(
                    {
                        "method": method,
                        "split": split_name,
                        "true_label": true_label,
                        "predicted_label": pred_label,
                        "count": int(matrix[true_idx, pred_idx]),
                    }
                )
    return pd.concat(reports, ignore_index=True), pd.DataFrame(confusions)


def plot_test_metrics(metrics: pd.DataFrame) -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / "ct_study_level_meta_test_metrics.png"
    plot_df = metrics[metrics["split"] == "test"].sort_values("f1_macro", ascending=True)
    fig, ax = plt.subplots(figsize=(9, max(4, 0.45 * len(plot_df))), dpi=160)
    y = np.arange(len(plot_df))
    ax.barh(y - 0.17, plot_df["f1_macro"], height=0.32, label="F1-macro", color="#2a9d8f")
    ax.barh(y + 0.17, plot_df["auc_roc_macro"], height=0.32, label="AUC macro", color="#457b9d")
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["method"], fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Valor")
    ax.set_title("CT study-level: reglas vs meta-clasificador")
    ax.grid(axis="x", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_best_confusion(predictions: pd.DataFrame, metrics: pd.DataFrame) -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / "ct_study_level_meta_best_confusion_matrix.png"
    best = metrics[metrics["split"] == "test"].sort_values(["f1_macro", "auc_roc_macro"], ascending=False).iloc[0]
    group = predictions[(predictions["split"] == "test") & (predictions["method"] == best["method"])]
    matrix = confusion_matrix(group["y_true"], group["y_pred"], labels=list(CLASS_NAMES))

    fig, ax = plt.subplots(figsize=(5, 4.5), dpi=160)
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(CLASS_NAMES)))
    ax.set_xticklabels([CLASS_NAMES[idx] for idx in CLASS_NAMES], rotation=35, ha="right")
    ax.set_yticks(range(len(CLASS_NAMES)))
    ax.set_yticklabels([CLASS_NAMES[idx] for idx in CLASS_NAMES])
    ax.set_xlabel("Prediccion")
    ax.set_ylabel("Etiqueta real")
    ax.set_title(f"Mejor metodo test: {best['method']}")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", color="black")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def write_summary(
    base_experiment: str,
    metrics: pd.DataFrame,
    validation: pd.DataFrame,
    figure_paths: list[Path],
) -> None:
    best_test = metrics[metrics["split"] == "test"].sort_values(["f1_macro", "auc_roc_macro"], ascending=False).iloc[0]
    best_val = validation.iloc[0]
    lines = [
        "# CT study-level meta-classifier",
        "",
        f"Base slice model: `{base_experiment}`.",
        "",
        "Esta fase usa las probabilidades de los slices informativos para construir una representacion por estudio.",
        "El objetivo es reducir el ruido metodologico de tratar cada slice como si tuviera una etiqueta perfecta.",
        "",
        "## Seleccion del meta-clasificador",
        "",
        f"- Mejor candidato en validacion: `{best_val['method']}`.",
        f"- F1-macro validacion: {best_val['f1_macro']:.4f}.",
        f"- AUC macro validacion: {best_val['auc_roc_macro']:.4f}.",
        "",
        "## Resultado test",
        "",
        f"- Mejor metodo test: `{best_test['method']}`.",
        f"- Accuracy test: {best_test['accuracy']:.4f}.",
        f"- F1-macro test: {best_test['f1_macro']:.4f}.",
        f"- AUC macro test: {best_test['auc_roc_macro']:.4f}.",
        "",
        "## Figuras",
        "",
    ]
    lines.extend(f"- `{path.relative_to(PROJECT_ROOT)}`" for path in figure_paths)
    lines.extend(
        [
            "",
            "Lectura metodologica: si el meta-clasificador supera a la media de probabilidades, indica que resumir el volumen con estadisticos por estudio aporta informacion adicional. Si no la supera, el resultado sigue siendo util porque muestra que la agregacion simple ya captura casi todo lo que el modelo 2D puede extraer.",
        ]
    )
    (RESULTS_DIR / "ct_study_level_meta_summary.md").write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    seed_everything(config.RANDOM_SEED)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    summary = load_summary(args.base_experiment)
    splits = split_metadata(summary)
    device = get_device()
    print(f"device={device}", flush=True)
    print(f"base_experiment={args.base_experiment}", flush=True)

    slice_predictions = collect_slice_predictions(
        summary=summary,
        splits=splits,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        force=args.force_inference,
    )
    study_features = make_study_feature_rows(slice_predictions)
    study_features.to_csv(RESULTS_DIR / "ct_study_level_features.csv", index=False)

    rule_metrics, rule_predictions = evaluate_rule_aggregations(study_features)
    validation_metrics, meta_test_metrics, meta_test_predictions = train_meta_models(study_features)

    metrics = pd.concat([pd.DataFrame(rule_metrics), meta_test_metrics], ignore_index=True)
    predictions = pd.concat([*rule_predictions, meta_test_predictions], ignore_index=True)
    reports, confusions = build_reports_and_confusions(predictions)

    metrics.to_csv(RESULTS_DIR / "ct_study_level_meta_metrics.csv", index=False)
    validation_metrics.to_csv(RESULTS_DIR / "ct_study_level_meta_validation_candidates.csv", index=False)
    predictions.to_csv(RESULTS_DIR / "ct_study_level_meta_predictions.csv", index=False)
    reports.to_csv(RESULTS_DIR / "ct_study_level_meta_classification_reports.csv", index=False)
    confusions.to_csv(RESULTS_DIR / "ct_study_level_meta_confusion_matrices.csv", index=False)

    metric_figure = plot_test_metrics(metrics)
    confusion_figure = plot_best_confusion(predictions, metrics)
    write_summary(args.base_experiment, metrics, validation_metrics, [metric_figure, confusion_figure])

    print("\nStudy-level meta metrics:")
    print(
        metrics[metrics["split"] == "test"][
            ["method", "n_studies", "accuracy", "f1_macro", "f1_weighted", "auc_roc_macro"]
        ]
        .sort_values(["f1_macro", "auc_roc_macro"], ascending=False)
        .round(4)
    )
    print(f"\nSaved metrics: {RESULTS_DIR / 'ct_study_level_meta_metrics.csv'}")
    print(f"Saved validation candidates: {RESULTS_DIR / 'ct_study_level_meta_validation_candidates.csv'}")
    print(f"Saved figure: {metric_figure}")
    print(f"Saved figure: {confusion_figure}")


if __name__ == "__main__":
    main()
