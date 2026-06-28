import json
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / "results" / ".matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OUTPUT_DIR = PROJECT_ROOT / "results" / "calibration"
FIGURES_DIR = OUTPUT_DIR / "figures"
N_BINS = 10

CLASS_NAMES = {
    "cxr": {0: "COVID", 1: "Lung Opacity", 2: "Normal", 3: "Viral Pneumonia"},
    "ct": {0: "CT-0", 1: "CT-1", 2: "CT-2", 3: "CT-3+"},
}


def read_json(path: Path) -> Dict:
    return json.loads(path.read_text())


def collect_classification_summaries() -> pd.DataFrame:
    rows = []
    for summary_path in sorted((PROJECT_ROOT / "results" / "classification").glob("*/*_full_summary.json")):
        summary = read_json(summary_path)
        predictions_path = summary_path.with_name(summary_path.name.replace("_summary.json", "_predictions.csv"))
        if not predictions_path.exists():
            continue
        rows.append(
            {
                **summary,
                "summary_path": str(summary_path),
                "predictions_path": str(predictions_path),
            }
        )
    return pd.DataFrame(rows).sort_values(["dataset", "accuracy", "f1_macro"], ascending=[True, False, False])


def probability_columns(predictions: pd.DataFrame) -> List[str]:
    return sorted([column for column in predictions.columns if column.startswith("prob_")], key=lambda name: int(name[5:]))


def expected_calibration_error(bin_df: pd.DataFrame, total_count: int) -> float:
    if total_count == 0 or bin_df.empty:
        return 0.0
    weighted_gap = bin_df["count"] * (bin_df["accuracy"] - bin_df["mean_confidence"]).abs()
    return float(weighted_gap.sum() / total_count)


def multiclass_brier_score(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    one_hot = np.zeros_like(probabilities)
    one_hot[np.arange(len(y_true)), y_true] = 1.0
    return float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))


def negative_log_likelihood(y_true: np.ndarray, probabilities: np.ndarray, eps: float = 1e-12) -> float:
    true_probs = np.clip(probabilities[np.arange(len(y_true)), y_true], eps, 1.0)
    return float(-np.mean(np.log(true_probs)))


def calibration_bins(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    confidences: np.ndarray,
    n_bins: int = N_BINS,
) -> pd.DataFrame:
    correct = y_true == y_pred
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    for idx in range(n_bins):
        lower = edges[idx]
        upper = edges[idx + 1]
        if idx == 0:
            mask = (confidences >= lower) & (confidences <= upper)
        else:
            mask = (confidences > lower) & (confidences <= upper)
        count = int(mask.sum())
        if count:
            mean_confidence = float(confidences[mask].mean())
            accuracy = float(correct[mask].mean())
        else:
            mean_confidence = float((lower + upper) / 2)
            accuracy = np.nan
        rows.append(
            {
                "bin": idx + 1,
                "bin_lower": float(lower),
                "bin_upper": float(upper),
                "bin_mid": float((lower + upper) / 2),
                "count": count,
                "mean_confidence": mean_confidence,
                "accuracy": accuracy,
                "abs_gap": float(abs(accuracy - mean_confidence)) if count else np.nan,
            }
        )
    return pd.DataFrame(rows)


def analyze_predictions(summary_row: pd.Series) -> Tuple[Dict, pd.DataFrame, pd.DataFrame]:
    predictions = pd.read_csv(summary_row["predictions_path"])
    prob_cols = probability_columns(predictions)
    probabilities = predictions[prob_cols].to_numpy(dtype=float)
    y_true = predictions["y_true"].to_numpy(dtype=int)
    y_pred = predictions["y_pred"].to_numpy(dtype=int)
    confidences = probabilities.max(axis=1)
    true_probabilities = probabilities[np.arange(len(y_true)), y_true]
    correct = y_true == y_pred

    bins = calibration_bins(y_true, y_pred, confidences)
    non_empty_bins = bins[bins["count"] > 0]
    ece = expected_calibration_error(non_empty_bins, len(y_true))
    mce = float(non_empty_bins["abs_gap"].max()) if not non_empty_bins.empty else 0.0
    high_conf_mask = confidences >= 0.90
    error_mask = ~correct
    high_conf_error_mask = high_conf_mask & error_mask

    metrics = {
        "experiment": summary_row["experiment"],
        "dataset": summary_row["dataset"],
        "architecture": summary_row["architecture"],
        "balance_strategy": summary_row["balance_strategy"],
        "accuracy": float(summary_row["accuracy"]),
        "f1_macro": float(summary_row["f1_macro"]),
        "auc_roc_macro": float(summary_row["auc_roc_macro"]) if pd.notna(summary_row.get("auc_roc_macro")) else np.nan,
        "n_samples": int(len(y_true)),
        "mean_confidence": float(confidences.mean()),
        "median_confidence": float(np.median(confidences)),
        "mean_true_class_probability": float(true_probabilities.mean()),
        "ece": ece,
        "mce": mce,
        "brier_score": multiclass_brier_score(y_true, probabilities),
        "negative_log_likelihood": negative_log_likelihood(y_true, probabilities),
        "high_confidence_count": int(high_conf_mask.sum()),
        "high_confidence_error_count": int(high_conf_error_mask.sum()),
        "high_confidence_error_rate": (
            float(high_conf_error_mask.sum() / high_conf_mask.sum()) if high_conf_mask.sum() else 0.0
        ),
        "error_count": int(error_mask.sum()),
        "mean_error_confidence": float(confidences[error_mask].mean()) if error_mask.any() else 0.0,
    }

    bins.insert(0, "experiment", summary_row["experiment"])
    bins.insert(1, "dataset", summary_row["dataset"])

    errors = predictions.loc[error_mask, ["y_true", "y_pred"]].copy()
    errors["experiment"] = summary_row["experiment"]
    errors["dataset"] = summary_row["dataset"]
    errors["confidence"] = confidences[error_mask]
    errors["true_class_probability"] = true_probabilities[error_mask]
    errors["y_true_label"] = errors["y_true"].map(CLASS_NAMES.get(summary_row["dataset"], {}))
    errors["y_pred_label"] = errors["y_pred"].map(CLASS_NAMES.get(summary_row["dataset"], {}))
    errors = errors[
        [
            "dataset",
            "experiment",
            "y_true",
            "y_true_label",
            "y_pred",
            "y_pred_label",
            "confidence",
            "true_class_probability",
        ]
    ].sort_values("confidence", ascending=False)
    return metrics, bins, errors


def write_dataframe(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def select_representative_models(summary_df: pd.DataFrame) -> pd.DataFrame:
    selected = []
    for dataset, group in summary_df.groupby("dataset"):
        by_accuracy = group.sort_values(["accuracy", "f1_macro"], ascending=False).iloc[0]
        selected.append(by_accuracy)
        by_f1 = group.sort_values(["f1_macro", "accuracy"], ascending=False).iloc[0]
        if by_f1["experiment"] != by_accuracy["experiment"]:
            selected.append(by_f1)
    return pd.DataFrame(selected).drop_duplicates(subset=["experiment"]).reset_index(drop=True)


def plot_reliability_diagrams(bins_df: pd.DataFrame, selected_experiments: Iterable[str]) -> Path:
    path = FIGURES_DIR / "reliability_diagrams_selected_models.png"
    selected_experiments = list(selected_experiments)
    n = len(selected_experiments)
    fig, axes = plt.subplots(1, n, figsize=(5.4 * n, 5), dpi=160, squeeze=False)
    for axis, experiment in zip(axes[0], selected_experiments):
        group = bins_df[(bins_df["experiment"] == experiment) & (bins_df["count"] > 0)].copy()
        dataset = group["dataset"].iloc[0] if not group.empty else ""
        axis.plot([0, 1], [0, 1], linestyle="--", color="0.45", label="Calibracion ideal")
        axis.bar(
            group["bin_mid"],
            group["accuracy"],
            width=0.085,
            alpha=0.75,
            edgecolor="black",
            label="Accuracy por bin",
        )
        axis.scatter(group["mean_confidence"], group["accuracy"], color="#b00020", s=28, zorder=3)
        axis.set_title(f"{dataset.upper()} | {experiment.replace(dataset + '_', '')}", fontsize=9)
        axis.set_xlabel("Confianza media")
        axis.set_ylabel("Accuracy observada")
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.grid(alpha=0.25)
    axes[0][0].legend(loc="upper left", fontsize=8)
    fig.suptitle("Reliability diagrams: confianza vs accuracy observada")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_confidence_histograms(summary_df: pd.DataFrame, selected_df: pd.DataFrame) -> Path:
    path = FIGURES_DIR / "confidence_histograms_selected_models.png"
    n = len(selected_df)
    fig, axes = plt.subplots(1, n, figsize=(5.4 * n, 4.5), dpi=160, squeeze=False)
    for axis, (_, row) in zip(axes[0], selected_df.iterrows()):
        predictions = pd.read_csv(row["predictions_path"])
        probabilities = predictions[probability_columns(predictions)].to_numpy(dtype=float)
        confidences = probabilities.max(axis=1)
        correct = predictions["y_true"].to_numpy(dtype=int) == predictions["y_pred"].to_numpy(dtype=int)
        axis.hist(confidences[correct], bins=np.linspace(0, 1, 11), alpha=0.75, label="Correctas")
        axis.hist(confidences[~correct], bins=np.linspace(0, 1, 11), alpha=0.75, label="Errores")
        axis.set_title(f"{row['dataset'].upper()} | {row['experiment'].replace(row['dataset'] + '_', '')}", fontsize=9)
        axis.set_xlabel("Confianza maxima")
        axis.set_ylabel("N")
        axis.set_xlim(0, 1)
        axis.grid(axis="y", alpha=0.25)
    axes[0][0].legend(loc="upper left", fontsize=8)
    fig.suptitle("Distribucion de confianza: aciertos vs errores")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_ece_by_experiment(metrics_df: pd.DataFrame) -> Path:
    path = FIGURES_DIR / "ece_by_experiment.png"
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), dpi=160)
    for axis, dataset in zip(axes, ["cxr", "ct"]):
        group = metrics_df[metrics_df["dataset"] == dataset].sort_values("ece")
        labels = group["experiment"].str.replace(f"{dataset}_", "", regex=False)
        y = np.arange(len(group))
        axis.barh(y, group["ece"], color="#4c78a8")
        axis.set_yticks(y)
        axis.set_yticklabels(labels, fontsize=7)
        axis.set_xlabel("ECE")
        axis.set_title(dataset.upper())
        axis.grid(axis="x", alpha=0.25)
    fig.suptitle("Expected Calibration Error por experimento")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def write_markdown_summary(metrics_df: pd.DataFrame, selected_df: pd.DataFrame, output_path: Path) -> None:
    lines = [
        "# Resumen de calibracion",
        "",
        "La calibracion evalua si la confianza probabilistica del clasificador refleja su accuracy observada.",
        "Un modelo bien calibrado deberia acertar aproximadamente el 90% de las muestras a las que asigna 0.90 de confianza.",
        "",
        "## Modelos representativos",
        "",
    ]
    for _, selected in selected_df.iterrows():
        row = metrics_df[metrics_df["experiment"] == selected["experiment"]].iloc[0]
        lines.extend(
            [
                f"### {row['dataset'].upper()} - `{row['experiment']}`",
                "",
                f"- Accuracy: `{row['accuracy']:.4f}`",
                f"- F1-macro: `{row['f1_macro']:.4f}`",
                f"- Confianza media: `{row['mean_confidence']:.4f}`",
                f"- ECE: `{row['ece']:.4f}`",
                f"- Brier score: `{row['brier_score']:.4f}`",
                f"- Errores con confianza >= 0.90: `{int(row['high_confidence_error_count'])}` de `{int(row['high_confidence_count'])}` predicciones de alta confianza",
                "",
            ]
        )

    best_calibrated = metrics_df.sort_values(["dataset", "ece"]).groupby("dataset").head(1)
    lines.extend(["## Lectura metodologica", ""])
    for _, row in best_calibrated.iterrows():
        lines.append(
            f"- En {row['dataset'].upper()}, el menor ECE lo obtiene `{row['experiment']}` con ECE `{row['ece']:.4f}`."
        )
    lines.extend(
        [
            "",
            "La calibracion no sustituye accuracy, F1 o AUC: complementa esas metricas indicando si las probabilidades del modelo son fiables.",
            "En contexto medico, los errores de alta confianza son especialmente relevantes porque indican predicciones incorrectas que el modelo presenta como seguras.",
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines))


def main() -> Dict[str, str]:
    summary_df = collect_classification_summaries()
    if summary_df.empty:
        raise FileNotFoundError("No classification full summaries with predictions were found.")

    metric_rows = []
    bin_frames = []
    error_frames = []
    for _, row in summary_df.iterrows():
        metrics, bins, errors = analyze_predictions(row)
        metric_rows.append(metrics)
        bin_frames.append(bins)
        if not errors.empty:
            error_frames.append(errors)

    metrics_df = pd.DataFrame(metric_rows).sort_values(["dataset", "ece", "accuracy"], ascending=[True, True, False])
    bins_df = pd.concat(bin_frames, ignore_index=True)
    errors_df = pd.concat(error_frames, ignore_index=True) if error_frames else pd.DataFrame()
    selected_df = select_representative_models(summary_df)

    write_dataframe(metrics_df, OUTPUT_DIR / "calibration_metrics.csv")
    write_dataframe(bins_df, OUTPUT_DIR / "calibration_bins.csv")
    write_dataframe(errors_df.head(100), OUTPUT_DIR / "high_confidence_errors_top100.csv")
    write_dataframe(selected_df, OUTPUT_DIR / "selected_models.csv")

    figures = {
        "reliability_diagrams": plot_reliability_diagrams(bins_df, selected_df["experiment"]),
        "confidence_histograms": plot_confidence_histograms(summary_df, selected_df),
        "ece_by_experiment": plot_ece_by_experiment(metrics_df),
    }
    write_markdown_summary(metrics_df, selected_df, OUTPUT_DIR / "calibration_summary.md")

    output = {
        "metrics": str(OUTPUT_DIR / "calibration_metrics.csv"),
        "bins": str(OUTPUT_DIR / "calibration_bins.csv"),
        "errors": str(OUTPUT_DIR / "high_confidence_errors_top100.csv"),
        "selected_models": str(OUTPUT_DIR / "selected_models.csv"),
        "summary": str(OUTPUT_DIR / "calibration_summary.md"),
        **{name: str(path) for name, path in figures.items()},
    }
    print(json.dumps(output, indent=2))
    return output


if __name__ == "__main__":
    main()
