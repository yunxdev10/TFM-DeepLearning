#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PROJECT_ROOT}/.conda/bin/python"

MODE="${1:-best}"
ROWS_PER_GROUP="${2:-3}"
BEST_EXPERIMENT="ct_attention_unet_mixed30_patch192_pos70_tversky_pos10_bf32_thr095_segmentation"

cd "${PROJECT_ROOT}"

if [[ "${MODE}" == "list" ]]; then
  "${PYTHON_BIN}" scripts/generate_ct_segmentation_visualizations.py --list-experiments
  exit 0
elif [[ "${MODE}" == "all" ]]; then
  "${PYTHON_BIN}" scripts/generate_ct_segmentation_visualizations.py \
    --all-ct \
    --run-mode full \
    --rows-per-group "${ROWS_PER_GROUP}"
else
  EXPERIMENT="${MODE}"
  if [[ "${MODE}" == "best" ]]; then
    EXPERIMENT="${BEST_EXPERIMENT}"
  fi

  "${PYTHON_BIN}" scripts/generate_ct_segmentation_visualizations.py \
    --experiment "${EXPERIMENT}" \
    --run-mode full \
    --rows-per-group "${ROWS_PER_GROUP}"
fi

echo
echo "Visualizaciones generadas en:"
echo "${PROJECT_ROOT}/results/segmentation/ct/qualitative"
