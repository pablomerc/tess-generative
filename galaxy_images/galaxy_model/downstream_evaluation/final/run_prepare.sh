#!/bin/bash
# ==============================================================================
# Prepare all downstream embeddings (real + untrained + random) for one checkpoint.
# Reproducibility: config matches prepare_hsc_provabgs.py lines 84-85
#   (zdim16_nogeom_neighbors, g2g9kvr4/checkpoints/latest-step=step=75000.ckpt)
# ==============================================================================

set -e

# ------------------------------------------------------------------------------
# Configuration (matches prepare_hsc_provabgs.py zdim16_nogeom_neighbors)
# ------------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="${SCRIPT_DIR}/prepare_all.py"

CHECKPOINT_PATH="/data/vision/billf/scratch/pablomer/projects/tess-generative/galaxy-flow-matching-neighbours/g2g9kvr4/checkpoints/latest-step=step=75000.ckpt"
MODULE_NAME="double_train_fm_neighbors.py"
SUFFIX="zdim16_nogeom_neighbors"

# Output: H5s written here (default: same dir as script so predict_all finds them)
OUTPUT_DIR="${SCRIPT_DIR}"

# Reproducibility
SEED=42
BATCH_SIZE=256
DATASETS="mmu,legacy_provabgs,neighbors"

# ------------------------------------------------------------------------------
# Run (execute from galaxy_model/ so imports resolve)
# ------------------------------------------------------------------------------
GALAXY_MODEL_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${GALAXY_MODEL_DIR}"

echo "----------------------------------------------------------------"
echo "Prepare all downstream embeddings (reproducible)"
echo "----------------------------------------------------------------"
echo "Script:     ${SCRIPT_PATH}"
echo "Checkpoint: ${CHECKPOINT_PATH}"
echo "Module:     ${MODULE_NAME}"
echo "Suffix:     ${SUFFIX}"
echo "Output:     ${OUTPUT_DIR}"
echo "Seed:       ${SEED}"
echo "----------------------------------------------------------------"

mkdir -p "$OUTPUT_DIR"

python "${SCRIPT_PATH}" \
    --checkpoint "$CHECKPOINT_PATH" \
    --module "$MODULE_NAME" \
    --suffix "$SUFFIX" \
    --output-dir "$OUTPUT_DIR" \
    --batch-size "$BATCH_SIZE" \
    --seed "$SEED" \
    --datasets "$DATASETS"

echo "----------------------------------------------------------------"
echo "Done. H5s: downstream_mmu_${SUFFIX}.h5, downstream_legacy_provabgs_${SUFFIX}.h5, downstream_neighbors_${SUFFIX}.h5"
echo "Run: python downstream_evaluation/final/predict_all.py --suffix ${SUFFIX} --output-dir ${OUTPUT_DIR}"
echo "----------------------------------------------------------------"
