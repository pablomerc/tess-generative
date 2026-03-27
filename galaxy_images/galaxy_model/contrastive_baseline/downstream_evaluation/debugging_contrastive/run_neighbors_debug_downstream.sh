#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GALAXY_MODEL_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${GALAXY_MODEL_DIR}"

OUTPUT_DIR="${SCRIPT_DIR}/outputs"
SEED=42
BATCH_SIZE=256

mkdir -p "${OUTPUT_DIR}"

MODEL_NAMES=("shards" "batches")
MODEL_PATHS=(
  "/data/vision/billf/scratch/pablomer/outputs/contrastive_baseline/copies/shards-100ksteps.ckpt"
  "/data/vision/billf/scratch/pablomer/outputs/contrastive_baseline/copies/batches-100ksteps.ckpt"
)

for i in "${!MODEL_PATHS[@]}"; do
  NAME="${MODEL_NAMES[$i]}"
  CKPT="${MODEL_PATHS[$i]}"
  SUFFIX="neighbors_debug_${NAME}"

  echo "=============================================="
  echo "Running neighbors debug downstream for: ${NAME}"
  echo "Checkpoint: ${CKPT}"
  echo "Suffix:     ${SUFFIX}"

  python "${SCRIPT_DIR}/prepare_neighbors_debug_contrastive.py" \
    --checkpoint "${CKPT}" \
    --suffix "${SUFFIX}" \
    --output-dir "${OUTPUT_DIR}" \
    --batch-size "${BATCH_SIZE}" \
    --seed "${SEED}"

  python "${SCRIPT_DIR}/predict_neighbors_debug_contrastive.py" \
    --suffix "${SUFFIX}" \
    --output-dir "${OUTPUT_DIR}" \
    --seed "${SEED}"
done

echo "All neighbors debug runs completed."
