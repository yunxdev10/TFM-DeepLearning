import json
import math
import os
import sys
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / "results" / ".matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, roc_auc_score

from src.config import config
from src.data.ct_preprocessing import get_ct_dataframes


OUTPUT_DIR = PROJECT_ROOT / "results" / "classification" / "ct_study_level"
FIGURES_DIR = OUTPUT_DIR / "figures"

LABEL_MAP = {"CT-0": 0, "CT-1": 1, "CT-2": 2, "CT-3+": 3}
CLASS_NAMES = {value: key for key, value in LABEL_MAP.items()}
CLASS_ORDER = [CLASS_NAMES[idx] for idx in range(len(CLASS_NAMES))]
PROBABILITY_COLUMNS = [f"prob_{idx}" for idx in range(len(CLASS_NAMES))]


def read_json(path: Path) -> Dict:
    return json.loads(path.read_text())


def collect_ct_summaries() -> pd.DataFrame:
    rows = []
    for summary_path in sorted((PROJECT_ROOT / "results" / "classification" / "ct").glob("*_full_summary.json")):
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
    if not rows:
        raise FileNotFoundError("No se encontraron summaries/predicciones CT full en results/classification/ct.")
    return pd.DataFrame(rows).sort_values(["architecture", "balance_strategy"]).reset_index(drop=True)


def load_ct_test_dataframe() -> pd.DataFrame:
    metadata_path = config.CT_DIR / "processed_2d_slices" / "labels_metadata.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(f"No existe el metadata CT procesado: {metadata_path}")
    metadata_df = pd.read_csv(metadata_path)
    _, _, test_df = get_ct_dataframes(metadata_df, config.RANDOM_SEED)
    return test_df.reset_index(drop=True)


def enrich_slice_predictions(predictions: pd.DataFrame, test_df: pd.DataFrame, experiment: str) -> pd.DataFrame:
    if len(predictions) != len(test_df):
        raise ValueError(
            f"{experiment}: el CSV de predicciones tiene {len(predictions)} filas, "
            f"pero el test_df reconstruido tiene {len(test_df)}."
        )

    expected_true = test_df["label"].map(LABEL_MAP).to_numpy(dtype=int)
    observed_true = predictions["y_true"].to_numpy(dtype=int)
    if not np.array_equal(expected_true, observed_true):
        raise ValueError(
            f"{experiment}: y_true no coincide con las etiquetas reconstruidas del test split. "
            "No se puede unir por indice de forma segura."
        )

    enriched = test_df[["study_id", "image_path", "slice_index", "total_slices", "label"]].copy()
    enriched.insert(0, "experiment", experiment)
    for column in ["y_true", "y_pred", *PROBABILITY_COLUMNS]:
        enriched[column] = predictions[column].to_numpy()
    probabilities = enriched[PROBABILITY_COLUMNS].to_numpy(dtype=float)
    enriched["slice_confidence"] = probabilities.max(axis=1)
    enriched["slice_correct"] = enriched["y_true"] == enriched["y_pred"]
    enriched["y_true_label"] = enriched["y_true"].map(CLASS_NAMES)
    enriched["y_pred_label"] = enriched["y_pred"].map(CLASS_NAMES)
    return enriched


def normalize_scores(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    scores = np.clip(scores, 0.0, None)
    total = scores.sum()
    if total <= 0:
        return np.ones_like(scores) / len(scores)
    return scores / total


def aggregate_mean_probability(probabilities: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return probabilities.mean(axis=0)


def aggregate_confidence_weighted_mean(probabilities: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    confidences = probabilities.max(axis=1)
    if float(confidences.sum()) <= 0:
        return aggregate_mean_probability(probabilities, y_pred)
    weights = confidences / confidences.sum()
    return (probabilities * weights[:, None]).sum(axis=0)


def aggregate_max_probability(probabilities: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return probabilities.max(axis=0)


def aggregate_topk_mean_probability(probabilities: np.ndarray, y_pred: np.ndarray, k: int = 3) -> np.ndarray:
    k = max(1, min(k, len(probabilities)))
    sorted_probs = np.sort(probabilities, axis=0)
    return sorted_probs[-k:, :].mean(axis=0)


def aggregate_top_fraction_mean_probability(
    probabilities: np.ndarray,
    y_pred: np.ndarray,
    fraction: float = 0.20,
) -> np.ndarray:
    k = max(1, int(math.ceil(len(probabilities) * fraction)))
    return aggregate_topk_mean_probability(probabilities, y_pred, k=k)


def aggregate_majority_vote(probabilities: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    counts = np.bincount(y_pred.astype(int), minlength=len(CLASS_NAMES)).astype(float)
    if counts.sum() <= 0:
        return aggregate_mean_probability(probabilities, y_pred)

    # If there is a tie, add a tiny probability-based tie breaker without changing
    # the interpretation of the vector as vote proportions.
    mean_probs = probabilities.mean(axis=0)
    tied_max = counts == counts.max()
    if tied_max.sum() > 1:
        counts = counts + 1e-6 * mean_probs
    return counts / counts.sum()


AGGREGATION_FUNCTIONS: Dict[str, Callable[[np.ndarray, np.ndarray], np.ndarray]] = {
    "mean_probability": aggregate_mean_probability,
    "confidence_weighted_mean": aggregate_confidence_weighted_mean,
    "max_probability": aggregate_max_probability,
    "top3_mean_probability": lambda probabilities, y_pred: aggregate_topk_mean_probability(probabilities, y_pred, k=3),
    "top20pct_mean_probability": aggregate_top_fraction_mean_probability,
    "majority_vote": aggregate_majority_vote,
}

AGGREGATION_DESCRIPTIONS = {
    "mean_probability": "Promedio de probabilidades de todos los slices del estudio.",
    "confidence_weighted_mean": "Promedio ponderado, dando mas peso a slices donde el modelo esta mas seguro.",
    "max_probability": "Maximo score observado por clase en cualquier slice del estudio.",
    "top3_mean_probability": "Promedio de los 3 slices con mayor probabilidad para cada clase.",
    "top20pct_mean_probability": "Promedio del 20% de slices mas altos para cada clase.",
    "majority_vote": "Voto mayoritario de las clases predichas slice a slice.",
}


def aggregate_by_study(enriched: pd.DataFrame, aggregation_name: str) -> pd.DataFrame:
    aggregation_fn = AGGREGATION_FUNCTIONS[aggregation_name]
    rows = []
    for study_id, group in enriched.groupby("study_id", sort=True):
        probabilities = group[PROBABILITY_COLUMNS].to_numpy(dtype=float)
        y_pred = group["y_pred"].to_numpy(dtype=int)
        scores = normalize_scores(aggregation_fn(probabilities, y_pred))
        predicted_class = int(np.argmax(scores))
        true_class = int(group["y_true"].iloc[0])
        rows.append(
            {
                "experiment": group["experiment"].iloc[0],
                "aggregation": aggregation_name,
                "study_id": study_id,
                "y_true": true_class,
                "y_pred": predicted_class,
                "y_true_label": CLASS_NAMES[true_class],
                "y_pred_label": CLASS_NAMES[predicted_class],
                "n_slices": int(len(group)),
                "slice_accuracy_within_study": float((group["y_true"] == group["y_pred"]).mean()),
                "mean_slice_confidence": float(group["slice_confidence"].mean()),
                "max_slice_confidence": float(group["slice_confidence"].max()),
                "slice_vote_fraction_for_study_prediction": float((group["y_pred"] == predicted_class).mean()),
                **{f"study_prob_{idx}": float(scores[idx]) for idx in range(len(CLASS_NAMES))},
            }
        )
    return pd.DataFrame(rows)


def compute_metrics(predictions: pd.DataFrame, unit: str, summary_row: pd.Series | None = None) -> Dict:
    y_true = predictions["y_true"].to_numpy(dtype=int)
    y_pred = predictions["y_pred"].to_numpy(dtype=int)
    score_columns = [column for column in predictions.columns if column.startswith("study_prob_")]
    y_score = predictions[score_columns].to_numpy(dtype=float) if score_columns else None

    metrics = {
        "experiment": predictions["experiment"].iloc[0],
        "unit": unit,
        "aggregation": predictions["aggregation"].iloc[0] if "aggregation" in predictions.columns else "slice",
        "n_samples": int(len(predictions)),
        "n_studies": int(predictions["study_id"].nunique()) if "study_id" in predictions.columns else np.nan,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }
    if y_score is not None:
        try:
            metrics["auc_roc_macro"] = float(roc_auc_score(y_true, y_score, multi_class="ovr", average="macro"))
        except ValueError:
            metrics["auc_roc_macro"] = np.nan
    elif summary_row is not None:
        metrics["auc_roc_macro"] = (
            float(summary_row["auc_roc_macro"]) if pd.notna(summary_row.get("auc_roc_macro")) else np.nan
        )
    else:
        metrics["auc_roc_macro"] = np.nan
    return metrics


def build_confusion_rows(predictions: pd.DataFrame) -> pd.DataFrame:
    matrix = confusion_matrix(predictions["y_true"], predictions["y_pred"], labels=list(CLASS_NAMES))
    rows = []
    for true_idx, true_label in CLASS_NAMES.items():
        for pred_idx, pred_label in CLASS_NAMES.items():
            rows.append(
                {
                    "experiment": predictions["experiment"].iloc[0],
                    "unit": "study",
                    "aggregation": predictions["aggregation"].iloc[0],
                    "true_label": true_label,
                    "predicted_label": pred_label,
                    "count": int(matrix[true_idx, pred_idx]),
                }
            )
    return pd.DataFrame(rows)


def build_classification_report_rows(predictions: pd.DataFrame) -> pd.DataFrame:
    report = classification_report(
        predictions["y_true"],
        predictions["y_pred"],
        labels=list(CLASS_NAMES),
        target_names=CLASS_ORDER,
        zero_division=0,
        output_dict=True,
    )
    report_df = pd.DataFrame(report).T.reset_index(names="label")
    report_df.insert(0, "aggregation", predictions["aggregation"].iloc[0])
    report_df.insert(0, "experiment", predictions["experiment"].iloc[0])
    return report_df


def shorten_experiment(name: str) -> str:
    return name.replace("ct_", "").replace("_full", "")


def plot_slice_vs_study(metrics_df: pd.DataFrame) -> Path:
    path = FIGURES_DIR / "ct_slice_vs_study_f1_macro.png"
    slice_df = metrics_df[metrics_df["unit"] == "slice"].copy()
    study_df = metrics_df[
        (metrics_df["unit"] == "study") & (metrics_df["aggregation"] == "mean_probability")
    ].copy()
    merged = slice_df.merge(
        study_df,
        on="experiment",
        suffixes=("_slice", "_study"),
    ).sort_values("f1_macro_study", ascending=True)

    fig, axis = plt.subplots(figsize=(10, max(5, 0.45 * len(merged))), dpi=160)
    y_pos = np.arange(len(merged))
    axis.barh(y_pos - 0.18, merged["f1_macro_slice"], height=0.34, label="Slice-level")
    axis.barh(y_pos + 0.18, merged["f1_macro_study"], height=0.34, label="Study-level mean")
    axis.set_yticks(y_pos)
    axis.set_yticklabels([shorten_experiment(name) for name in merged["experiment"]], fontsize=8)
    axis.set_xlabel("F1-macro")
    axis.set_title("CT: comparacion entre evaluacion por slice y por estudio")
    axis.set_xlim(0, 1)
    axis.grid(axis="x", alpha=0.25)
    axis.legend(loc="lower right")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_aggregation_comparison(metrics_df: pd.DataFrame) -> Path:
    path = FIGURES_DIR / "ct_study_level_aggregation_comparison.png"
    study_df = metrics_df[metrics_df["unit"] == "study"].copy()
    best_per_aggregation = (
        study_df.sort_values(["aggregation", "f1_macro", "accuracy"], ascending=[True, False, False])
        .groupby("aggregation", as_index=False)
        .first()
        .sort_values("f1_macro", ascending=True)
    )

    fig, axis = plt.subplots(figsize=(9, 5), dpi=160)
    y_pos = np.arange(len(best_per_aggregation))
    axis.barh(y_pos - 0.18, best_per_aggregation["accuracy"], height=0.34, label="Accuracy")
    axis.barh(y_pos + 0.18, best_per_aggregation["f1_macro"], height=0.34, label="F1-macro")
    labels = [
        f"{row.aggregation}\n{shorten_experiment(row.experiment)}"
        for row in best_per_aggregation.itertuples(index=False)
    ]
    axis.set_yticks(y_pos)
    axis.set_yticklabels(labels, fontsize=8)
    axis.set_xlabel("Metrica")
    axis.set_title("CT por estudio: mejor modelo encontrado por estrategia de agregacion")
    axis.set_xlim(0, 1)
    axis.grid(axis="x", alpha=0.25)
    axis.legend(loc="lower right")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_best_confusion_matrix(predictions_df: pd.DataFrame, best_row: pd.Series) -> Path:
    path = FIGURES_DIR / "ct_study_level_best_confusion_matrix.png"
    subset = predictions_df[
        (predictions_df["experiment"] == best_row["experiment"])
        & (predictions_df["aggregation"] == best_row["aggregation"])
    ]
    matrix = confusion_matrix(subset["y_true"], subset["y_pred"], labels=list(CLASS_NAMES))

    fig, axis = plt.subplots(figsize=(6, 5), dpi=160)
    image = axis.imshow(matrix, cmap="Blues")
    axis.set_xticks(np.arange(len(CLASS_ORDER)))
    axis.set_yticks(np.arange(len(CLASS_ORDER)))
    axis.set_xticklabels(CLASS_ORDER, rotation=35, ha="right")
    axis.set_yticklabels(CLASS_ORDER)
    axis.set_xlabel("Prediccion")
    axis.set_ylabel("Etiqueta real")
    axis.set_title(
        "Mejor evaluacion CT por estudio\n"
        f"{shorten_experiment(best_row['experiment'])} | {best_row['aggregation']}"
    )
    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            axis.text(col_idx, row_idx, str(matrix[row_idx, col_idx]), ha="center", va="center", color="black")
    fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def write_summary(
    metrics_df: pd.DataFrame,
    test_df: pd.DataFrame,
    figure_paths: Iterable[Path],
) -> Path:
    summary_path = OUTPUT_DIR / "ct_study_level_summary.md"
    study_df = metrics_df[metrics_df["unit"] == "study"].copy()
    slice_df = metrics_df[metrics_df["unit"] == "slice"].copy()
    best_study = study_df.sort_values(["f1_macro", "accuracy"], ascending=False).iloc[0]
    best_slice = slice_df.sort_values(["f1_macro", "accuracy"], ascending=False).iloc[0]
    mean_rows = study_df[study_df["aggregation"] == "mean_probability"]
    best_mean = mean_rows.sort_values(["f1_macro", "accuracy"], ascending=False).iloc[0]

    lines = [
        "# Evaluacion CT por estudio",
        "",
        "Esta fase agrega las predicciones slice a slice de los clasificadores CT para obtener una unica prediccion por `study_id`.",
        "La motivacion es metodologica: MosMedData asigna la etiqueta CT-0/CT-1/CT-2/CT-3+ al estudio completo, no a cada corte individual.",
        "",
        "## Unidad de evaluacion",
        "",
        f"- Slices de test: {len(test_df)}",
        f"- Estudios de test: {test_df['study_id'].nunique()}",
        "- Split: reconstruido con `get_ct_dataframes` y la semilla del proyecto.",
        "- Seguridad metodologica: las predicciones se unen al test split por indice y se verifica que `y_true` coincida antes de agregar.",
        "",
        "## Estrategias de agregacion",
        "",
    ]
    for name, description in AGGREGATION_DESCRIPTIONS.items():
        lines.append(f"- `{name}`: {description}")
    lines.extend(
        [
            "",
            "## Mejor resultado por estudio",
            "",
            f"- Modelo: `{best_study['experiment']}`",
            f"- Agregacion: `{best_study['aggregation']}`",
            f"- Accuracy por estudio: {best_study['accuracy']:.4f}",
            f"- F1-macro por estudio: {best_study['f1_macro']:.4f}",
            f"- AUC-ROC macro por estudio: {best_study['auc_roc_macro']:.4f}",
            "",
            "## Comparacion con evaluacion por slice",
            "",
            f"- Mejor F1-macro slice-level: `{best_slice['experiment']}` = {best_slice['f1_macro']:.4f}",
            f"- Mejor F1-macro study-level usando `mean_probability`: `{best_mean['experiment']}` = {best_mean['f1_macro']:.4f}",
            "",
            "La evaluacion por estudio no sustituye la evaluacion por slice, sino que la complementa. "
            "Es mas coherente con la etiqueta original de MosMedData, pero reduce el numero de muestras de test y hace mas visible el desbalance por paciente.",
            "",
            "## Artefactos generados",
            "",
            "- `ct_study_level_metrics.csv`: metricas por modelo y estrategia de agregacion.",
            "- `ct_study_level_predictions.csv`: predicciones agregadas por estudio.",
            "- `ct_slice_predictions_with_study.csv`: predicciones por slice enriquecidas con `study_id`.",
            "- `ct_study_level_confusion_matrices.csv`: matrices de confusion en formato largo.",
            "- `ct_study_level_classification_reports.csv`: precision, recall y F1 por clase.",
        ]
    )
    for figure_path in figure_paths:
        lines.append(f"- `{figure_path.relative_to(PROJECT_ROOT)}`")
    lines.extend(
        [
            "",
            "## Lectura para la memoria",
            "",
            "La mejora principal de esta fase es conceptual: permite evaluar los clasificadores CT en la misma unidad en la que MosMedData define sus etiquetas. "
            "Al agregar los slices de un mismo estudio, se reduce el efecto de cortes aislados poco informativos. "
            "Aun asi, los resultados deben interpretarse con cautela porque el conjunto de test por estudio es mucho mas pequeno que el conjunto por slice y sigue estando desbalanceado.",
        ]
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text("\n".join(lines) + "\n")
    return summary_path


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    summaries_df = collect_ct_summaries()
    test_df = load_ct_test_dataframe()

    enriched_slice_frames = []
    study_prediction_frames = []
    metrics_rows = []
    confusion_frames = []
    report_frames = []

    for _, summary_row in summaries_df.iterrows():
        predictions = pd.read_csv(summary_row["predictions_path"])
        enriched = enrich_slice_predictions(predictions, test_df, summary_row["experiment"])
        enriched_slice_frames.append(enriched)
        metrics_rows.append(compute_metrics(enriched, unit="slice", summary_row=summary_row))

        for aggregation_name in AGGREGATION_FUNCTIONS:
            study_predictions = aggregate_by_study(enriched, aggregation_name)
            study_prediction_frames.append(study_predictions)
            metrics_rows.append(compute_metrics(study_predictions, unit="study"))
            confusion_frames.append(build_confusion_rows(study_predictions))
            report_frames.append(build_classification_report_rows(study_predictions))

    slice_predictions_df = pd.concat(enriched_slice_frames, ignore_index=True)
    study_predictions_df = pd.concat(study_prediction_frames, ignore_index=True)
    metrics_df = pd.DataFrame(metrics_rows).sort_values(
        ["unit", "f1_macro", "accuracy"],
        ascending=[True, False, False],
    )
    confusion_df = pd.concat(confusion_frames, ignore_index=True)
    reports_df = pd.concat(report_frames, ignore_index=True)

    slice_predictions_df.to_csv(OUTPUT_DIR / "ct_slice_predictions_with_study.csv", index=False)
    study_predictions_df.to_csv(OUTPUT_DIR / "ct_study_level_predictions.csv", index=False)
    metrics_df.to_csv(OUTPUT_DIR / "ct_study_level_metrics.csv", index=False)
    confusion_df.to_csv(OUTPUT_DIR / "ct_study_level_confusion_matrices.csv", index=False)
    reports_df.to_csv(OUTPUT_DIR / "ct_study_level_classification_reports.csv", index=False)

    best_study_row = metrics_df[metrics_df["unit"] == "study"].sort_values(
        ["f1_macro", "accuracy"], ascending=False
    ).iloc[0]
    figure_paths = [
        plot_slice_vs_study(metrics_df),
        plot_aggregation_comparison(metrics_df),
        plot_best_confusion_matrix(study_predictions_df, best_study_row),
    ]
    summary_path = write_summary(metrics_df, test_df, figure_paths)

    print(f"Saved metrics: {OUTPUT_DIR / 'ct_study_level_metrics.csv'}")
    print(f"Saved study predictions: {OUTPUT_DIR / 'ct_study_level_predictions.csv'}")
    print(f"Saved summary: {summary_path}")
    print("\nBest study-level rows:")
    display_cols = ["experiment", "unit", "aggregation", "n_samples", "accuracy", "f1_macro", "auc_roc_macro"]
    print(
        metrics_df[metrics_df["unit"] == "study"]
        .sort_values(["f1_macro", "accuracy"], ascending=False)
        .head(12)[display_cols]
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
