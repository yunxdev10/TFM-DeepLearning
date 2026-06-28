#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PROJECT_ROOT}/.conda/bin/python"

MODE="${1:-both}"
MAX_PER_CLASS="${2:-2}"
MAX_INCORRECT_PER_CLASS="${3:-1}"
CT_MASK_SPLIT="${4:-test}"
XAI_DEVICE="${XAI_DEVICE:-cpu}"

cd "${PROJECT_ROOT}"

"${PYTHON_BIN}" scripts/generate_xai_explanations.py \
  --dataset "${MODE}" \
  --run-mode full \
  --max-per-class "${MAX_PER_CLASS}" \
  --max-incorrect-per-class "${MAX_INCORRECT_PER_CLASS}" \
  --ct-mask-split "${CT_MASK_SPLIT}" \
  --device "${XAI_DEVICE}"

echo
echo "Resultados XAI generados en:"
echo "${PROJECT_ROOT}/results/explainability"
