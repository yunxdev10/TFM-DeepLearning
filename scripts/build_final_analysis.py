import json
import math
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
from sklearn.metrics import accuracy_score, f1_score


OUTPUT_DIR = PROJECT_ROOT / "results" / "final_analysis"
FIGURES_DIR = OUTPUT_DIR / "figures"


def read_json(path: Path) -> Dict:
    return json.loads(path.read_text())


def collect_classification_results() -> pd.DataFrame:
    rows = []
    for summary_path in sorted((PROJECT_ROOT / "results" / "classification").glob("*/*_full_summary.json")):
        summary = read_json(summary_path)
        predictions_path = summary_path.with_name(summary_path.name.replace("_summary.json", "_predictions.csv"))
        report_path = summary_path.with_name(summary_path.name.replace("_summary.json", "_classification_report.csv"))
        rows.append(
            {
                **summary,
                "summary_path": str(summary_path),
                "predictions_path": str(predictions_path) if predictions_path.exists() else None,
                "classification_report_path": str(report_path) if report_path.exists() else None,
            }
        )
    return pd.DataFrame(rows).sort_values(["dataset", "accuracy", "f1_macro"], ascending=[True, False, False])


def bootstrap_classification_ci(predictions_path: Path, n_bootstrap: int = 500, seed: int = 42) -> Dict[str, float]:
    predictions = pd.read_csv(predictions_path)
    y_true = predictions["y_true"].to_numpy()
    y_pred = predictions["y_pred"].to_numpy()
    rng = np.random.default_rng(seed)
    indices = np.arange(len(y_true))
    accuracy_values = []
    f1_values = []
    for _ in range(n_bootstrap):
        sample_indices = rng.choice(indices, size=len(indices), replace=True)
        accuracy_values.append(accuracy_score(y_true[sample_indices], y_pred[sample_indices]))
        f1_values.append(f1_score(y_true[sample_indices], y_pred[sample_indices], average="macro", zero_division=0))
    return {
        "accuracy_ci_low": float(np.quantile(accuracy_values, 0.025)),
        "accuracy_ci_high": float(np.quantile(accuracy_values, 0.975)),
        "f1_macro_ci_low": float(np.quantile(f1_values, 0.025)),
        "f1_macro_ci_high": float(np.quantile(f1_values, 0.975)),
    }


def add_classification_ci(classification_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in classification_df.iterrows():
        output = row.to_dict()
        if row.get("predictions_path"):
            output.update(bootstrap_classification_ci(Path(row["predictions_path"])))
        rows.append(output)
    return pd.DataFrame(rows)


def mcnemar_approx(predictions_a: Path, predictions_b: Path) -> Dict[str, float]:
    a = pd.read_csv(predictions_a)
    b = pd.read_csv(predictions_b)
    if len(a) != len(b) or not np.array_equal(a["y_true"].to_numpy(), b["y_true"].to_numpy()):
        return {"n01": math.nan, "n10": math.nan, "chi2": math.nan, "p_value": math.nan}

    y_true = a["y_true"].to_numpy()
    a_correct = a["y_pred"].to_numpy() == y_true
    b_correct = b["y_pred"].to_numpy() == y_true
    n01 = int((~a_correct & b_correct).sum())
    n10 = int((a_correct & ~b_correct).sum())
    denom = n01 + n10
    if denom == 0:
        return {"n01": n01, "n10": n10, "chi2": 0.0, "p_value": 1.0}
    chi2 = (abs(n01 - n10) - 1) ** 2 / denom
    p_value = math.erfc(math.sqrt(chi2 / 2))
    return {"n01": n01, "n10": n10, "chi2": float(chi2), "p_value": float(p_value)}


def build_mcnemar_results(classification_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset, group in classification_df.groupby("dataset"):
        top = group.sort_values(["accuracy", "f1_macro"], ascending=False).head(2).reset_index(drop=True)
        if len(top) < 2:
            continue
        stats = mcnemar_approx(Path(top.loc[0, "predictions_path"]), Path(top.loc[1, "predictions_path"]))
        rows.append(
            {
                "dataset": dataset,
                "model_a": top.loc[0, "experiment"],
                "model_b": top.loc[1, "experiment"],
                "accuracy_a": float(top.loc[0, "accuracy"]),
                "accuracy_b": float(top.loc[1, "accuracy"]),
                **stats,
            }
        )
    return pd.DataFrame(rows)


def collect_segmentation_results() -> pd.DataFrame:
    rows = []
    for summary_path in sorted((PROJECT_ROOT / "results" / "segmentation").glob("*/*_full_summary.json")):
        summary = read_json(summary_path)
        rows.append(
            {
                "experiment": summary.get("experiment"),
                "dataset": summary.get("dataset"),
                "run_mode": summary.get("run_mode"),
                "architecture": summary.get("architecture"),
                "variant_name": summary.get("variant_name"),
                "loss_name": summary.get("loss_name"),
                "dice": summary.get("dice"),
                "iou": summary.get("iou"),
                "pixel_accuracy": summary.get("pixel_accuracy"),
                "threshold": summary.get("threshold"),
                "summary_path": str(summary_path),
            }
        )
    return pd.DataFrame(rows).sort_values(["dataset", "dice", "iou"], ascending=[True, False, False])


def infer_xai_scope(path: Path) -> str:
    folder = path.parent.name
    if folder.endswith("_all"):
        return "all"
    if folder.endswith("_test"):
        return "test"
    return "test"


def collect_xai_results() -> pd.DataFrame:
    rows = []
    for summary_path in sorted((PROJECT_ROOT / "results" / "explainability").glob("*/*/*_xai_summary.json")):
        summary = read_json(summary_path)
        if summary.get("dataset") == "ct" and not summary_path.parent.name.endswith(("_all", "_test")):
            continue
        rows.append(
            {
                **summary,
                "mask_split": infer_xai_scope(summary_path),
                "summary_path": str(summary_path),
            }
        )
    return pd.DataFrame(rows).sort_values(["dataset", "experiment", "mask_split"])


def write_dataframe(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def plot_classification(classification_df: pd.DataFrame) -> Path:
    path = FIGURES_DIR / "classification_accuracy_f1_macro.png"
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), dpi=160, sharex=False)
    for axis, dataset in zip(axes, ["cxr", "ct"]):
        group = classification_df[classification_df["dataset"] == dataset].sort_values("accuracy")
        labels = group["experiment"].str.replace(f"{dataset}_", "", regex=False)
        y = np.arange(len(group))
        axis.barh(y - 0.18, group["accuracy"], height=0.35, label="Accuracy")
        axis.barh(y + 0.18, group["f1_macro"], height=0.35, label="F1 macro")
        axis.set_yticks(y)
        axis.set_yticklabels(labels, fontsize=7)
        axis.set_xlim(0, 1)
        axis.set_title(dataset.upper())
        axis.grid(axis="x", alpha=0.25)
    axes[0].legend(loc="lower right")
    fig.suptitle("Clasificacion: accuracy y F1-macro por experimento")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_segmentation(segmentation_df: pd.DataFrame) -> Path:
    path = FIGURES_DIR / "segmentation_dice_iou.png"
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), dpi=160, sharex=False)
    for axis, dataset in zip(axes, ["cxr", "ct"]):
        group = segmentation_df[segmentation_df["dataset"] == dataset].sort_values("dice")
        labels = group["experiment"].str.replace(f"{dataset}_", "", regex=False)
        y = np.arange(len(group))
        axis.barh(y - 0.18, group["dice"], height=0.35, label="Dice")
        axis.barh(y + 0.18, group["iou"], height=0.35, label="IoU")
        axis.set_yticks(y)
        axis.set_yticklabels(labels, fontsize=7)
        axis.set_xlim(0, 1)
        axis.set_title(dataset.upper())
        axis.grid(axis="x", alpha=0.25)
    axes[0].legend(loc="lower right")
    fig.suptitle("Segmentacion: Dice e IoU por experimento")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_xai(xai_df: pd.DataFrame) -> Path:
    path = FIGURES_DIR / "xai_gradcam_alignment.png"
    if xai_df.empty:
        return path
    labels = xai_df.apply(lambda row: f"{row['dataset']} | {row['experiment']} | {row['mask_split']}", axis=1)
    y = np.arange(len(xai_df))
    fig, axis = plt.subplots(figsize=(12, max(4, 0.55 * len(xai_df))), dpi=160)
    axis.barh(y - 0.18, xai_df["mean_saliency_mask_iou"], height=0.35, label="IoU saliencia-mascara")
    axis.barh(y + 0.18, xai_df["mean_saliency_inside_mask_ratio"], height=0.35, label="Ratio dentro mascara")
    axis.set_yticks(y)
    axis.set_yticklabels(labels, fontsize=8)
    axis.set_xlim(0, 1)
    axis.grid(axis="x", alpha=0.25)
    axis.set_title("XAI Grad-CAM: alineacion con mascaras disponibles")
    axis.legend(loc="lower right")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def best_rows(classification_df: pd.DataFrame, segmentation_df: pd.DataFrame, xai_df: pd.DataFrame) -> Dict[str, Dict]:
    best_classification_accuracy = (
        classification_df.sort_values(["dataset", "accuracy", "f1_macro"], ascending=[True, False, False])
        .groupby("dataset")
        .head(1)
        .set_index("dataset")
        .to_dict(orient="index")
    )
    best_classification_f1 = (
        classification_df.sort_values(["dataset", "f1_macro", "accuracy"], ascending=[True, False, False])
        .groupby("dataset")
        .head(1)
        .set_index("dataset")
        .to_dict(orient="index")
    )
    best_segmentation = (
        segmentation_df.sort_values(["dataset", "dice", "iou"], ascending=[True, False, False])
        .groupby("dataset")
        .head(1)
        .set_index("dataset")
        .to_dict(orient="index")
    )
    xai_summary = {}
    if not xai_df.empty:
        for dataset, group in xai_df.groupby("dataset"):
            xai_summary[dataset] = {
                "best_iou_experiment": group.sort_values("mean_saliency_mask_iou", ascending=False).iloc[0].to_dict(),
                "mean_iou": float(group["mean_saliency_mask_iou"].mean()),
            }
    return {
        "classification_best_accuracy": best_classification_accuracy,
        "classification_best_f1": best_classification_f1,
        "segmentation_best_dice": best_segmentation,
        "xai_gradcam": xai_summary,
    }


def format_pct(value: float) -> str:
    return f"{100 * value:.1f}%"


def build_rq_summary(
    classification_df: pd.DataFrame,
    segmentation_df: pd.DataFrame,
    xai_df: pd.DataFrame,
    mcnemar_df: pd.DataFrame,
) -> str:
    best = best_rows(classification_df, segmentation_df, xai_df)
    cxr_cls = best["classification_best_f1"]["cxr"]
    ct_acc = best["classification_best_accuracy"]["ct"]
    ct_f1 = best["classification_best_f1"]["ct"]
    cxr_seg = best["segmentation_best_dice"]["cxr"]
    ct_seg = best["segmentation_best_dice"]["ct"]
    cxr_xai = xai_df[xai_df["dataset"] == "cxr"].sort_values("mean_saliency_mask_iou", ascending=False).iloc[0]
    ct_xai = xai_df[xai_df["dataset"] == "ct"].sort_values("mean_saliency_mask_iou", ascending=False).iloc[0]

    lines = [
        "# Resumen final por preguntas de investigacion",
        "",
        "## RQ1 - Arquitecturas de clasificacion",
        f"- En CXR, el mejor modelo global es `{cxr_cls['experiment']}` con accuracy `{cxr_cls['accuracy']:.4f}`, F1-macro `{cxr_cls['f1_macro']:.4f}` y AUC macro `{cxr_cls['auc_roc_macro']:.4f}`.",
        f"- En CT, el mejor modelo por accuracy es `{ct_acc['experiment']}` con accuracy `{ct_acc['accuracy']:.4f}`; por F1-macro/AUC destaca `{ct_f1['experiment']}` con F1-macro `{ct_f1['f1_macro']:.4f}` y AUC macro `{ct_f1['auc_roc_macro']:.4f}`.",
        "",
        "## RQ2 - Comparacion cross-modal",
        f"- CXR alcanza un rendimiento claramente superior a CT: mejor F1-macro CXR `{cxr_cls['f1_macro']:.4f}` frente a mejor F1-macro CT `{ct_f1['f1_macro']:.4f}`.",
        "- La diferencia apoya que CT de severidad es una tarea mas dificil por etiquetas ordinales, desbalance y variabilidad por slice.",
        "",
        "## RQ3 - Explicabilidad",
        f"- En CXR, Grad-CAM frente a mascara pulmonar obtiene IoU medio `{cxr_xai['mean_saliency_mask_iou']:.4f}` y ratio dentro de mascara `{cxr_xai['mean_saliency_inside_mask_ratio']:.4f}`.",
        f"- En CT, Grad-CAM frente a mascara de infeccion obtiene IoU maximo observado `{ct_xai['mean_saliency_mask_iou']:.4f}` y pico dentro de mascara `{ct_xai['peak_inside_mask_rate']:.4f}`.",
        "- La lectura clave es que CXR muestra plausibilidad anatomica parcial, mientras que CT no demuestra atencion localizada en lesion.",
        "",
        "## RQ4 - Segmentacion",
        f"- En CXR, el mejor resultado es `{cxr_seg['experiment']}` con Dice `{cxr_seg['dice']:.4f}` e IoU `{cxr_seg['iou']:.4f}`.",
        f"- En CT, el mejor resultado es `{ct_seg['experiment']}` con Dice `{ct_seg['dice']:.4f}` e IoU `{ct_seg['iou']:.4f}`.",
        "",
        "## RQ5 - Desbalanceo",
        "- En CXR, weighted cross-entropy mejora el mejor resultado global.",
        "- En CT, las estrategias de balanceo no superan claramente al baseline; el desbalance sigue siendo una limitacion importante.",
        "",
        "## Evidencia estadistica",
    ]
    if mcnemar_df.empty:
        lines.append("- No se generaron comparaciones McNemar.")
    else:
        for _, row in mcnemar_df.iterrows():
            lines.append(
                f"- {row['dataset'].upper()}: `{row['model_a']}` vs `{row['model_b']}` "
                f"p≈`{row['p_value']:.4g}` (n01={int(row['n01'])}, n10={int(row['n10'])})."
            )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    classification_df = collect_classification_results()
    classification_ci_df = add_classification_ci(classification_df)
    segmentation_df = collect_segmentation_results()
    xai_df = collect_xai_results()
    mcnemar_df = build_mcnemar_results(classification_ci_df)

    write_dataframe(classification_ci_df, OUTPUT_DIR / "classification_results_with_ci.csv")
    write_dataframe(
        classification_ci_df.sort_values(["dataset", "accuracy", "f1_macro"], ascending=[True, False, False])
        .groupby("dataset")
        .head(1),
        OUTPUT_DIR / "classification_best_by_accuracy.csv",
    )
    write_dataframe(
        classification_ci_df.sort_values(["dataset", "f1_macro", "accuracy"], ascending=[True, False, False])
        .groupby("dataset")
        .head(1),
        OUTPUT_DIR / "classification_best_by_f1.csv",
    )
    write_dataframe(mcnemar_df, OUTPUT_DIR / "classification_mcnemar_top2.csv")
    write_dataframe(segmentation_df, OUTPUT_DIR / "segmentation_results.csv")
    write_dataframe(
        segmentation_df.sort_values(["dataset", "dice", "iou"], ascending=[True, False, False])
        .groupby("dataset")
        .head(1),
        OUTPUT_DIR / "segmentation_best_by_dice.csv",
    )
    write_dataframe(xai_df, OUTPUT_DIR / "xai_gradcam_results.csv")

    figures = {
        "classification": str(plot_classification(classification_ci_df)),
        "segmentation": str(plot_segmentation(segmentation_df)),
        "xai": str(plot_xai(xai_df)),
    }
    summary = {
        "best": best_rows(classification_ci_df, segmentation_df, xai_df),
        "figures": figures,
        "generated_files": sorted(str(path) for path in OUTPUT_DIR.rglob("*") if path.is_file()),
    }
    (OUTPUT_DIR / "final_analysis_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    rq_summary = build_rq_summary(classification_ci_df, segmentation_df, xai_df, mcnemar_df)
    (OUTPUT_DIR / "rq_summary.md").write_text(rq_summary)

    print("Final analysis artifacts generated:")
    for path in sorted(OUTPUT_DIR.rglob("*")):
        if path.is_file():
            print(path)


if __name__ == "__main__":
    main()
