#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GALAXY_MODEL_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${GALAXY_MODEL_DIR}"

OUTPUT_DIR="${SCRIPT_DIR}/outputs"
BATCH_SIZE=256
MODULE_FILE="double_train_fm_neighbors.py"
CHECKPOINT_PATH="/data/vision/billf/scratch/pablomer/projects/tess-generative/galaxy-flow-matching-neighbours/g2g9kvr4/checkpoints/latest-step=step=75000.ckpt"
SUFFIX="ours_debug_neighbors_75k"
SEED=42

mkdir -p "${OUTPUT_DIR}"

echo "=============================================="
echo "Running neighbors debug downstream for OURS"
echo "Checkpoint: ${CHECKPOINT_PATH}"
echo "Module:     ${MODULE_FILE}"
echo "Suffix:     ${SUFFIX}"
echo "Output dir: ${OUTPUT_DIR}"

python "${SCRIPT_DIR}/prepare_neighbors_ours_debug.py" \
  --checkpoint "${CHECKPOINT_PATH}" \
  --module "${MODULE_FILE}" \
  --suffix "${SUFFIX}" \
  --output-dir "${OUTPUT_DIR}" \
  --batch-size "${BATCH_SIZE}"

python "${SCRIPT_DIR}/predict_neighbors_ours_debug.py" \
  --suffix "${SUFFIX}" \
  --output-dir "${OUTPUT_DIR}" \
  --seed "${SEED}"

echo "OURS neighbors debug run completed."
