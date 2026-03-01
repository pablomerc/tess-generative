#!/bin/bash
# ==============================================================================
# Run predict_all.py on H5s produced by run_prepare.sh.
# Uses same SUFFIX and OUTPUT_DIR so it finds downstream_*_SUFFIX.h5.
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUFFIX="zdim16_nogeom_neighbors"
OUTPUT_DIR="${SCRIPT_DIR}"
SEED=42

GALAXY_MODEL_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${GALAXY_MODEL_DIR}"

echo "----------------------------------------------------------------"
echo "Predict all (real / untrained / random)"
echo "----------------------------------------------------------------"
echo "Suffix:     ${SUFFIX}"
echo "Output:     ${OUTPUT_DIR}"
echo "Seed:       ${SEED}"
echo "----------------------------------------------------------------"

python downstream_evaluation/final/predict_all.py \
    --suffix "$SUFFIX" \
    --output-dir "$OUTPUT_DIR" \
    --seed "$SEED"

echo "----------------------------------------------------------------"
echo "Done. Results: ${OUTPUT_DIR}/predict_all_${SUFFIX}.csv"
echo "----------------------------------------------------------------"
