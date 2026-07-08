#!/bin/bash
# Smoke test for the instrument -> physics MLP baseline.
# Run this from inside an existing Slurm allocation (salloc / sbatch / srun)
# on a compute node. It picks GPU if visible, otherwise CPU.
#
# Usage:  bash run_smoke_instrument_to_physics.sh

set -eo pipefail

SCRIPT_DIR="/home/pablomer/orcd/pool/tess-generative/galaxy_images/galaxy_model/downstream_evaluation/engaging"
OUT_CSV="${SCRIPT_DIR}/outputs/predictions/predict_instrument_baseline_smoke.csv"

if [[ "$(hostname)" == login* ]]; then
    echo "ERROR: you're on a login node ($(hostname)). Get a compute node first:"
    echo "  salloc -p mit_normal_gpu --gres=gpu:1 --cpus-per-task=4 --mem=16G --time=00:30:00"
    echo "then re-run this script from inside the allocation."
    exit 1
fi

source ~/.bashrc
conda activate torchenv
export TORCH_BLAS_PREFER_HIPBLASLT=0
export OMP_NUM_THREADS=4

if nvidia-smi -L >/dev/null 2>&1; then
    DEVICE=cuda
    echo "GPU detected: $(nvidia-smi -L | head -1)"
else
    DEVICE=cpu
    echo "No GPU visible; falling back to CPU (this dataset is tiny so it's still fast)."
fi

mkdir -p "${SCRIPT_DIR}/outputs/predictions"
cd "${SCRIPT_DIR}"

python predict_instrument_to_physics.py \
    --labels-source outputs/embeddings/base.h5 \
    --out "${OUT_CSV}" \
    --device "${DEVICE}" \
    --max-epochs 10 \
    --seed 0

echo ""
echo "=== ${OUT_CSV} ==="
cat "${OUT_CSV}"
