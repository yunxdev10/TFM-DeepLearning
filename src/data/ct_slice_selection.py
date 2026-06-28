from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from PIL import Image


@dataclass(frozen=True)
class SliceSelectionVariant:
    name: str
    description: str
    method: str
    lower_position: float | None = None
    upper_position: float | None = None
    top_k: int | None = None
    score_column: str = "gt10_frac"


SLICE_SELECTION_VARIANTS: dict[str, SliceSelectionVariant] = {
    "central30_70": SliceSelectionVariant(
        name="central30_70",
        description="Conserva slices en la zona central 30%-70% del volumen.",
        method="position",
        lower_position=0.30,
        upper_position=0.70,
    ),
    "central35_65": SliceSelectionVariant(
        name="central35_65",
        description="Conserva slices en la zona central 35%-65% del volumen.",
        method="position",
        lower_position=0.35,
        upper_position=0.65,
    ),
    "top12_tissue": SliceSelectionVariant(
        name="top12_tissue",
        description="Conserva los 12 slices con mayor cobertura tisular por estudio.",
        method="top_k",
        top_k=12,
    ),
    "top16_tissue": SliceSelectionVariant(
        name="top16_tissue",
        description="Conserva los 16 slices con mayor cobertura tisular por estudio.",
        method="top_k",
        top_k=16,
    ),
    "top20_tissue": SliceSelectionVariant(
        name="top20_tissue",
        description="Conserva los 20 slices con mayor cobertura tisular por estudio.",
        method="top_k",
        top_k=20,
    ),
}


QUALITY_COLUMNS = [
    "slice_position",
    "mean_intensity",
    "std_intensity",
    "nonzero_frac",
    "gt5_frac",
    "gt10_frac",
    "gt25_frac",
]


def load_ct_metadata(metadata_path: Path) -> pd.DataFrame:
    metadata = pd.read_csv(metadata_path)
    required_columns = {"study_id", "image_path", "slice_index", "total_slices", "label"}
    missing = required_columns.difference(metadata.columns)
    if missing:
        raise ValueError(f"CT metadata missing required columns: {sorted(missing)}")
    return metadata.reset_index(drop=True)


def compute_slice_quality_features(metadata: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in metadata.itertuples(index=False):
        image = np.asarray(Image.open(row.image_path).convert("L"), dtype=np.uint8)
        total_slices = max(int(row.total_slices) - 1, 1)
        rows.append(
            {
                "study_id": row.study_id,
                "image_path": row.image_path,
                "slice_index": int(row.slice_index),
                "total_slices": int(row.total_slices),
                "label": row.label,
                "slice_position": float(row.slice_index / total_slices),
                "mean_intensity": float(image.mean()),
                "std_intensity": float(image.std()),
                "nonzero_frac": float((image > 0).mean()),
                "gt5_frac": float((image > 5).mean()),
                "gt10_frac": float((image > 10).mean()),
                "gt25_frac": float((image > 25).mean()),
            }
        )
    return pd.DataFrame(rows)


def load_or_build_slice_quality_features(
    metadata_path: Path,
    cache_path: Path,
    force: bool = False,
) -> pd.DataFrame:
    if cache_path.exists() and not force:
        cached = pd.read_csv(cache_path)
        if set(QUALITY_COLUMNS).issubset(cached.columns):
            return cached.reset_index(drop=True)

    metadata = load_ct_metadata(metadata_path)
    quality = compute_slice_quality_features(metadata)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    quality.to_csv(cache_path, index=False)
    return quality.reset_index(drop=True)


def select_informative_slices(
    quality_df: pd.DataFrame,
    variant: SliceSelectionVariant,
) -> pd.DataFrame:
    if variant.method == "position":
        if variant.lower_position is None or variant.upper_position is None:
            raise ValueError(f"Position variant {variant.name!r} needs lower and upper bounds.")
        selected = quality_df[
            quality_df["slice_position"].between(variant.lower_position, variant.upper_position, inclusive="both")
        ].copy()
    elif variant.method == "top_k":
        if variant.top_k is None:
            raise ValueError(f"Top-k variant {variant.name!r} needs top_k.")
        selected = (
            quality_df.sort_values(
                ["study_id", variant.score_column, "std_intensity", "mean_intensity"],
                ascending=[True, False, False, False],
            )
            .groupby("study_id", group_keys=False)
            .head(variant.top_k)
            .sort_values(["study_id", "slice_index"])
            .copy()
        )
    else:
        raise ValueError(f"Unknown slice selection method: {variant.method!r}")

    return selected.reset_index(drop=True)


def summarize_slice_selection(base_df: pd.DataFrame, selected_df: pd.DataFrame, variant_name: str) -> pd.DataFrame:
    base_counts = _counts_by_label(base_df, "base")
    selected_counts = _counts_by_label(selected_df, "selected")
    summary = base_counts.merge(selected_counts, on="label", how="outer").fillna(0)
    summary.insert(0, "variant", variant_name)
    summary["slice_keep_ratio"] = summary["selected_slices"] / summary["base_slices"].replace(0, np.nan)
    summary["study_keep_ratio"] = summary["selected_studies"] / summary["base_studies"].replace(0, np.nan)
    return summary.sort_values("label").reset_index(drop=True)


def _counts_by_label(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    slice_counts = df["label"].value_counts().rename(f"{prefix}_slices")
    study_counts = df[["study_id", "label"]].drop_duplicates()["label"].value_counts().rename(f"{prefix}_studies")
    return pd.concat([slice_counts, study_counts], axis=1).rename_axis("label").reset_index()


def assert_studies_preserved(base_df: pd.DataFrame, selected_df: pd.DataFrame, variant_name: str) -> None:
    base_studies = set(base_df["study_id"].unique())
    selected_studies = set(selected_df["study_id"].unique())
    missing = sorted(base_studies.difference(selected_studies))
    if missing:
        preview = ", ".join(missing[:10])
        raise ValueError(
            f"Variant {variant_name!r} removed {len(missing)} complete studies. "
            f"First missing studies: {preview}"
        )


def prepare_selection_metadata(
    metadata_path: Path,
    output_dir: Path,
    variants: Iterable[str] | None = None,
    force_quality: bool = False,
) -> tuple[pd.DataFrame, dict[str, Path]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    quality_path = output_dir / "ct_slice_quality_features.csv"
    quality_df = load_or_build_slice_quality_features(metadata_path, quality_path, force=force_quality)

    variant_names = list(variants) if variants is not None else list(SLICE_SELECTION_VARIANTS)
    summary_frames = []
    metadata_paths: dict[str, Path] = {}
    for variant_name in variant_names:
        variant = SLICE_SELECTION_VARIANTS[variant_name]
        selected = select_informative_slices(quality_df, variant)
        assert_studies_preserved(quality_df, selected, variant_name)
        selected = selected[["study_id", "image_path", "slice_index", "total_slices", "label", *QUALITY_COLUMNS]]
        selected_path = output_dir / f"{variant_name}_metadata.csv"
        selected.to_csv(selected_path, index=False)
        metadata_paths[variant_name] = selected_path
        summary_frames.append(summarize_slice_selection(quality_df, selected, variant_name))

    summary = pd.concat(summary_frames, ignore_index=True)
    summary.to_csv(output_dir / "ct_slice_selection_summary.csv", index=False)
    return summary, metadata_paths
