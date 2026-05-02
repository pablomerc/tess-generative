#!/bin/bash
# Smoke test: runs all 3 stages on 512 galaxies interactively (~2-3 min with GPU).
# Edit CHECKPOINT and SUFFIX below, then: bash smoke_test.sh

set -eo pipefail

CHECKPOINT="/work1/jeroenaudenaert/pablomer/tess-generative/galaxy_images/galaxy_model/outputs/neighbors_all_attn/2026-04-05_2/checkpoints/latest-step=step=104000.ckpt"
SUFFIX="test"

# -----------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd /work1/jeroenaudenaert/pablomer/tess-generative
source ~/.bashrc
conda activate torchenv
export PYTHONPATH=/work1/jeroenaudenaert/pablomer/tess-generative:$PYTHONPATH
export TORCH_BLAS_PREFER_HIPBLASLT=0
export OMP_NUM_THREADS=1

echo "=== Stage 1: prepare (512 galaxies) ==="
python "${SCRIPT_DIR}/prepare_hsc_downstream.py" \
    --checkpoint "${CHECKPOINT}" \
    --suffix     "${SUFFIX}"     \
    --output-dir "${SCRIPT_DIR}" \
    --num-examples 512

echo ""
echo "=== Stage 2: predict ==="
python "${SCRIPT_DIR}/predict_hsc_downstream.py" \
    --suffix     "${SUFFIX}"     \
    --output-dir "${SCRIPT_DIR}"

echo ""
echo "=== Stage 3: polished plot ==="
python "${SCRIPT_DIR}/compare_hsc.py" \
    --suffix     "${SUFFIX}"     \
    --output-dir "${SCRIPT_DIR}"

echo ""
echo "Done. Outputs in ${SCRIPT_DIR}:"
ls -lh "${SCRIPT_DIR}/downstream_hsc_new_${SUFFIX}.h5" \
       "${SCRIPT_DIR}/predict_hsc_downstream_${SUFFIX}.csv" \
       "${SCRIPT_DIR}/compare_hsc_${SUFFIX}.png" 2>/dev/null || true
