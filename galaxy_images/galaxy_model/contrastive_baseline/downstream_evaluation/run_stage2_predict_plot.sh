#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GALAXY_MODEL_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${GALAXY_MODEL_DIR}"

# ---- Edit this to match Stage 1 ----
SUFFIX="contrastive_latest"
# ------------------------------------

OUTPUT_DIR="${SCRIPT_DIR}"
SEED=42

echo "Stage 2 (predict + plots) starting..."
echo "Suffix:     ${SUFFIX}"
echo "Output dir: ${OUTPUT_DIR}"

python "${SCRIPT_DIR}/predict_all_contrastive.py" \
  --suffix "${SUFFIX}" \
  --output-dir "${OUTPUT_DIR}" \
  --seed "${SEED}"

echo "Stage 2 done."
