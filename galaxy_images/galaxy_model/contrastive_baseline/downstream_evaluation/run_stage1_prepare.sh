#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GALAXY_MODEL_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${GALAXY_MODEL_DIR}"

# ---- Edit these ----
# CHECKPOINT_PATH="/data/vision/billf/scratch/pablomer/outputs/contrastive_baseline/best-epoch=11-step=17000.ckpt"
CHECKPOINT_PATH='/data/vision/billf/scratch/pablomer/outputs/contrastive_baseline/best-epoch=77-step=120000.ckpt'

SUFFIX="contrastive_latest"
# --------------------

OUTPUT_DIR="${SCRIPT_DIR}"
SEED=42
BATCH_SIZE=256
DATASETS="mmu,legacy_provabgs,neighbors,hsc_provabgs"

echo "Stage 1 (prepare) starting..."
echo "Checkpoint: ${CHECKPOINT_PATH}"
echo "Suffix:     ${SUFFIX}"
echo "Output dir: ${OUTPUT_DIR}"

python "${SCRIPT_DIR}/prepare_all_contrastive.py" \
  --checkpoint "${CHECKPOINT_PATH}" \
  --suffix "${SUFFIX}" \
  --output-dir "${OUTPUT_DIR}" \
  --batch-size "${BATCH_SIZE}" \
  --seed "${SEED}" \
  --datasets "${DATASETS}"

echo "Stage 1 done."
