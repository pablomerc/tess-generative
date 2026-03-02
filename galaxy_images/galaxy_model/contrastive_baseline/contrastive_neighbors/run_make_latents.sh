#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GALAXY_MODEL_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${GALAXY_MODEL_DIR}"

# ---- Edit these ----
CHECKPOINT_PATH="/data/vision/billf/scratch/pablomer/outputs/contrastive_baseline/best-epoch=11-step=17000.ckpt"
SUFFIX="contrastive_baseline"
# --------------------

MAX_EXAMPLES=200000
BATCH_SIZE=256
OUTPUT_PATH="${SCRIPT_DIR}/contrastive_neighbor_latents_${SUFFIX}.h5"

# Force a stable runtime environment on shared nodes.
export CUDA_VISIBLE_DEVICES=0
export MPLCONFIGDIR=/tmp/matplotlib
export NUMBA_CACHE_DIR=/tmp/numba_cache
mkdir -p "${MPLCONFIGDIR}" "${NUMBA_CACHE_DIR}"

echo "----------------------------------------------------------------"
echo "Make contrastive neighbor latents"
echo "----------------------------------------------------------------"
echo "Checkpoint:   ${CHECKPOINT_PATH}"
echo "Suffix:       ${SUFFIX}"
echo "Max examples: ${MAX_EXAMPLES}"
echo "Output:       ${OUTPUT_PATH}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "----------------------------------------------------------------"

python "${SCRIPT_DIR}/make_latents_all.py" \
  --checkpoint "${CHECKPOINT_PATH}" \
  --suffix "${SUFFIX}" \
  --output "${OUTPUT_PATH}" \
  --max-examples "${MAX_EXAMPLES}" \
  --batch-size "${BATCH_SIZE}"

echo "----------------------------------------------------------------"
echo "Done. Run:"
echo "python ${SCRIPT_DIR}/search_neighbors.py --latents ${OUTPUT_PATH} --index 0"
echo "----------------------------------------------------------------"
