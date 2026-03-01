#!/bin/bash
# ==============================================================================
# Build neighbor latent file (physics + instrument embeddings) for kNN search.
# Default checkpoint matches downstream_evaluation/final/run_prepare.sh
# ==============================================================================

set -e

# ------------------------------------------------------------------------------
# Configuration (same checkpoint as run_prepare.sh)
# ------------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="${SCRIPT_DIR}/make_latents_all.py"

CHECKPOINT_PATH="/data/vision/billf/scratch/pablomer/projects/tess-generative/galaxy-flow-matching-neighbours/g2g9kvr4/checkpoints/latest-step=step=75000.ckpt"
MODULE_NAME="double_train_fm_neighbors.py"
SUFFIX="zdim16_nogeom_neighbors"

MAX_EXAMPLES=200000
BATCH_SIZE=256

# Output H5 (default: neighbor_search/neighbor_latents_<suffix>.h5)
OUTPUT_PATH="${SCRIPT_DIR}/neighbor_latents_${SUFFIX}.h5"

# ------------------------------------------------------------------------------
# Run (execute from galaxy_model/ so imports resolve)
# ------------------------------------------------------------------------------
GALAXY_MODEL_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${GALAXY_MODEL_DIR}"

echo "----------------------------------------------------------------"
echo "Make neighbor latents (physics + instrument embeddings)"
echo "----------------------------------------------------------------"
echo "Script:       ${SCRIPT_PATH}"
echo "Checkpoint:   ${CHECKPOINT_PATH}"
echo "Module:       ${MODULE_NAME}"
echo "Suffix:       ${SUFFIX}"
echo "Max examples: ${MAX_EXAMPLES}"
echo "Output:       ${OUTPUT_PATH}"
echo "----------------------------------------------------------------"

python "${SCRIPT_PATH}" \
    --checkpoint "$CHECKPOINT_PATH" \
    --module "$MODULE_NAME" \
    --suffix "$SUFFIX" \
    --output "$OUTPUT_PATH" \
    --max-examples "$MAX_EXAMPLES" \
    --batch-size "$BATCH_SIZE"

echo "----------------------------------------------------------------"
echo "Done. Run: python neighbor_search/search_neighbors.py --latents ${OUTPUT_PATH} --index 0"
echo "----------------------------------------------------------------"
