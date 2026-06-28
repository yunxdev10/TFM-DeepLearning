#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

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


RESULTS_DIR = PROJECT_ROOT / "results" / "classification" / "ct_informative_slices"
FIGURES_DIR = RESULTS_DIR / "figures"
LABEL_MAP = {"CT-0": 0, "CT-1": 1, "CT-2": 2, "CT-3+": 3}
CLASS_NAMES = {value: key for key, value in LABEL_MAP.items()}
PROBABILITY_COLUMNS = [f"prob_{idx}" for idx in range(len(CLASS_NAMES))]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def collect_summaries() -> pd.DataFrame:
    rows = []
    for summary_path in sorted(RESULTS_DIR.glob("*_full_summary.json")):
        summary = read_json(summary_path)
        predictions_path = summary_path.with_name(summary_path.name.replace("_summary.json", "_predictions.csv"))
        if not predictions_path.exists() or "slice_selection_variant" not in summary:
            continue
        rows.append({**summary, "summary_path": str(summary_path), "predictions_path": str(predictions_path)})
    return pd.DataFrame(rows)


def enrich_predictions(summary: pd.Series) -> pd.DataFrame:
    metadata_df = pd.read_csv(summary["metadata_path"])
    _, _, test_df = get_ct_dataframes(metadata_df, config.RANDOM_SEED)
    predictions = pd.read_csv(summary["predictions_path"])
    if len(predictions) != len(test_df):
        raise ValueError(
            f"{summary['experiment']}: predictions has {len(predictions)} rows but reconstructed test has {len(test_df)}."
        )
    expected = test_df["label"].map(LABEL_MAP).to_numpy(dtype=int)
    observed = predictions["y_true"].to_numpy(dtype=int)
    if not np.array_equal(expected, observed):
        raise ValueError(f"{summary['experiment']}: y_true does not match reconstructed CT test split.")

    enriched = test_df[["study_id", "image_path", "slice_index", "total_slices", "label"]].reset_index(drop=True)
    enriched.insert(0, "experiment", summary["experiment"])
    enriched.insert(1, "slice_selection_variant", summary["slice_selection_variant"])
    for column in ["y_true", "y_pred", *PROBABILITY_COLUMNS]:
        enriched[column] = predictions[column].to_numpy()
    enriched["slice_confidence"] = enriched[PROBABILITY_COLUMNS].to_numpy(dtype=float).max(axis=1)
    return enriched


def normalize(scores: np.ndarray) -> np.ndarray:
    scores = np.clip(np.asarray(scores, dtype=float), 0.0, None)
    total = scores.sum()
    if total <= 0:
        return np.ones_like(scores) / len(scores)
    return scores / total


def aggregate_by_study(enriched: pd.DataFrame, aggregation: str) -> pd.DataFrame:
    rows = []
    for study_id, group in enriched.groupby("study_id", sort=True):
        probabilities = group[PROBABILITY_COLUMNS].to_numpy(dtype=float)
        if aggregation == "mean_probability":
            scores = probabilities.mean(axis=0)
        elif aggregation == "confidence_weighted_mean":
            weights = group["slice_confidence"].to_numpy(dtype=float)
            weights = weights / weights.sum() if weights.sum() > 0 else np.ones(len(group)) / len(group)
            scores = (probabilities * weights[:, None]).sum(axis=0)
        elif aggregation == "majority_vote":
            votes = np.bincount(group["y_pred"].to_numpy(dtype=int), minlength=len(CLASS_NAMES)).astype(float)
            scores = votes / votes.sum()
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")

        scores = normalize(scores)
        true_class = int(group["y_true"].iloc[0])
        pred_class = int(np.argmax(scores))
        rows.append(
            {
                "experiment": group["experiment"].iloc[0],
                "slice_selection_variant": group["slice_selection_variant"].iloc[0],
                "aggregation": aggregation,
                "study_id": study_id,
                "n_slices": int(len(group)),
                "y_true": true_class,
                "y_pred": pred_class,
                "y_true_label": CLASS_NAMES[true_class],
                "y_pred_label": CLASS_NAMES[pred_class],
                **{f"study_prob_{idx}": float(scores[idx]) for idx in range(len(CLASS_NAMES))},
            }
        )
    return pd.DataFrame(rows)


def metric_row(predictions: pd.DataFrame, unit: str, aggregation: str = "slice") -> dict:
    y_true = predictions["y_true"].to_numpy(dtype=int)
    y_pred = predictions["y_pred"].to_numpy(dtype=int)
    prob_columns = [column for column in predictions.columns if column.startswith("study_prob_")]
    if not prob_columns:
        prob_columns = PROBABILITY_COLUMNS if set(PROBABILITY_COLUMNS).issubset(predictions.columns) else []
    y_prob = predictions[prob_columns].to_numpy(dtype=float) if prob_columns else None
    try:
        auc = float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")) if y_prob is not None else np.nan
    except ValueError:
        auc = np.nan
    return {
        "experiment": predictions["experiment"].iloc[0],
        "slice_selection_variant": predictions["slice_selection_variant"].iloc[0],
        "unit": unit,
        "aggregation": aggregation,
        "n_samples": int(len(predictions)),
        "n_studies": int(predictions["study_id"].nunique()) if "study_id" in predictions.columns else np.nan,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "auc_roc_macro": auc,
    }


def report_rows(predictions: pd.DataFrame, unit: str, aggregation: str) -> pd.DataFrame:
    report = classification_report(
        predictions["y_true"],
        predictions["y_pred"],
        labels=list(CLASS_NAMES),
        target_names=[CLASS_NAMES[idx] for idx in CLASS_NAMES],
        zero_division=0,
        output_dict=True,
    )
    report_df = pd.DataFrame(report).T.reset_index(names="label")
    report_df.insert(0, "aggregation", aggregation)
    report_df.insert(0, "unit", unit)
    report_df.insert(0, "slice_selection_variant", predictions["slice_selection_variant"].iloc[0])
    report_df.insert(0, "experiment", predictions["experiment"].iloc[0])
    return report_df


def confusion_rows(predictions: pd.DataFrame, unit: str, aggregation: str) -> pd.DataFrame:
    matrix = confusion_matrix(predictions["y_true"], predictions["y_pred"], labels=list(CLASS_NAMES))
    rows = []
    for true_idx, true_label in CLASS_NAMES.items():
        for pred_idx, pred_label in CLASS_NAMES.items():
            rows.append(
                {
                    "experiment": predictions["experiment"].iloc[0],
                    "slice_selection_variant": predictions["slice_selection_variant"].iloc[0],
                    "unit": unit,
                    "aggregation": aggregation,
                    "true_label": true_label,
                    "predicted_label": pred_label,
                    "count": int(matrix[true_idx, pred_idx]),
                }
            )
    return pd.DataFrame(rows)


def plot_metrics(metrics: pd.DataFrame) -> Path:
    path = FIGURES_DIR / "ct_informative_slice_f1_macro.png"
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    if metrics.empty:
        return path

    plot_df = metrics.sort_values(["unit", "f1_macro"], ascending=[True, True]).copy()
    plot_df["label"] = plot_df.apply(
        lambda row: f"{row['experiment'].replace('ct_', '')}\\n{row['unit']} | {row['aggregation']}",
        axis=1,
    )
    fig, ax = plt.subplots(figsize=(12, max(5, 0.35 * len(plot_df))), dpi=160)
    y = np.arange(len(plot_df))
    ax.barh(y, plot_df["f1_macro"], color=np.where(plot_df["unit"].eq("study"), "#2a9d8f", "#457b9d"))
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["label"], fontsize=7)
    ax.set_xlim(0, 1)
    ax.set_xlabel("F1-macro")
    ax.set_title("CT: seleccion de slices informativos")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summaries = collect_summaries()
    if summaries.empty:
        print("No hay experimentos CT de seleccion de slices entrenados todavia.")
        print("Ejecuta primero: .conda/bin/python scripts/run_ct_informative_slice_experiments.py train-one --variant top16_tissue --architecture resnet50 --strategy weighted_ce")
        return

    all_enriched = []
    all_study = []
    metrics = []
    reports = []
    confusions = []
    for _, summary in summaries.iterrows():
        enriched = enrich_predictions(summary)
        all_enriched.append(enriched)
        metrics.append(metric_row(enriched, unit="slice"))
        reports.append(report_rows(enriched, unit="slice", aggregation="slice"))
        confusions.append(confusion_rows(enriched, unit="slice", aggregation="slice"))

        for aggregation in ("mean_probability", "confidence_weighted_mean", "majority_vote"):
            study_df = aggregate_by_study(enriched, aggregation)
            all_study.append(study_df)
            metrics.append(metric_row(study_df, unit="study", aggregation=aggregation))
            reports.append(report_rows(study_df, unit="study", aggregation=aggregation))
            confusions.append(confusion_rows(study_df, unit="study", aggregation=aggregation))

    enriched_df = pd.concat(all_enriched, ignore_index=True)
    study_predictions_df = pd.concat(all_study, ignore_index=True)
    metrics_df = pd.DataFrame(metrics).sort_values(["unit", "f1_macro", "accuracy"], ascending=[True, False, False])
    report_df = pd.concat(reports, ignore_index=True)
    confusion_df = pd.concat(confusions, ignore_index=True)

    enriched_df.to_csv(RESULTS_DIR / "ct_informative_slice_predictions_with_study.csv", index=False)
    study_predictions_df.to_csv(RESULTS_DIR / "ct_informative_slice_study_predictions.csv", index=False)
    metrics_df.to_csv(RESULTS_DIR / "ct_informative_slice_metrics.csv", index=False)
    report_df.to_csv(RESULTS_DIR / "ct_informative_slice_classification_reports.csv", index=False)
    confusion_df.to_csv(RESULTS_DIR / "ct_informative_slice_confusion_matrices.csv", index=False)
    figure_path = plot_metrics(metrics_df)

    print("\nInformative-slice metrics:")
    print(metrics_df[["experiment", "unit", "aggregation", "n_samples", "n_studies", "accuracy", "f1_macro", "auc_roc_macro"]].round(4))
    print(f"\nSaved figure: {figure_path}")


if __name__ == "__main__":
    main()
