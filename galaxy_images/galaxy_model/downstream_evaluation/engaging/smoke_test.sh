#!/bin/bash
# End-to-end smoke test on tiny subsets, base checkpoint only.
# Runs in ~5 min on a single GPU.

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
CKPT="${REPO_ROOT}/galaxy_images/galaxy_model/checkpoints/base/snapshot.ckpt"

source ~/.bashrc
conda activate torchenv
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export TORCH_BLAS_PREFER_HIPBLASLT=0
export OMP_NUM_THREADS=1

cd "${REPO_ROOT}"

OUT="${SCRIPT_DIR}/outputs/smoke"
mkdir -p "${OUT}/index" "${OUT}/embeddings" "${OUT}/predictions" "${OUT}/plots"

echo "=== [1/4] build_overlap_dataset (cap 512) ==="
python "${SCRIPT_DIR}/build_overlap_dataset.py" \
    --hsc-dir    /work1/jeroenaudenaert/pablomer/data/hsc_downstream \
    --legacy-dir /work1/jeroenaudenaert/pablomer/data/legacy_downstream_quick \
    --out-dir    "${OUT}/index" \
    --max-per-subset 512 \
    --match-radius-arcsec 1.0 \
    --seed 42

echo ""
echo "=== [2/4] prepare_combined (base, --smoke) ==="
python "${SCRIPT_DIR}/prepare_combined.py" \
    --checkpoint   "${CKPT}" \
    --module       double_train_fm_neighbors.py \
    --model-class  ConditionalFlowMatchingModule \
    --index-dir    "${OUT}/index" \
    --out          "${OUT}/embeddings/base.h5" \
    --batch-size   128 \
    --smoke

echo ""
echo "=== [3/4] predict_combined ==="
python "${SCRIPT_DIR}/predict_combined.py" \
    --embeddings "${OUT}/embeddings/base.h5" \
    --out        "${OUT}/predictions/predict_base.csv"

echo ""
echo "=== [4/4] makeplot ==="
python "${SCRIPT_DIR}/makeplot.py" \
    --predictions-dir "${OUT}/predictions" \
    --out-dir         "${OUT}/plots" \
    --checkpoints     base

echo ""
echo "Smoke artifacts under: ${OUT}"
ls -lh "${OUT}/index" "${OUT}/embeddings" "${OUT}/predictions" "${OUT}/plots" 2>/dev/null || true
