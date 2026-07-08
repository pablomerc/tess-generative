#!/bin/bash
#SBATCH --job-name=engaging-untrained
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=2:00:00
#SBATCH --partition=mit_normal_gpu
#SBATCH --mem=64G
#SBATCH --output=/orcd/pool/007/pablomer/tess-generative/galaxy_images/galaxy_model/downstream_evaluation/engaging/logs/%x_%j.out
#SBATCH --error=/orcd/pool/007/pablomer/tess-generative/galaxy_images/galaxy_model/downstream_evaluation/engaging/logs/%x_%j.err
#SBATCH --chdir=/orcd/pool/007/pablomer/tess-generative
#SBATCH --open-mode=append

# One-off: encode the overlap dataset with a random-init ResNet-18 and probe it.
# Output CSV (predict_untrained-resnet.csv) is reused as the "Untrained ResNet-18"
# reference series in every per-variant publication plot.

set -eo pipefail

SCRIPT_DIR="/home/pablomer/orcd/pool/tess-generative/galaxy_images/galaxy_model/downstream_evaluation/engaging"
REPO_ROOT="/home/pablomer/orcd/pool/tess-generative"

source ~/.bashrc
conda activate torchenv
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export TORCH_BLAS_PREFER_HIPBLASLT=0
export OMP_NUM_THREADS=1
export WANDB_MODE=disabled

cd "${REPO_ROOT}"

mkdir -p "${SCRIPT_DIR}/outputs/embeddings" "${SCRIPT_DIR}/outputs/predictions"

H5="${SCRIPT_DIR}/outputs/embeddings/untrained-resnet.h5"
CSV="${SCRIPT_DIR}/outputs/predictions/predict_untrained-resnet.csv"

if [[ ! -f "${H5}" ]]; then
    echo "=== Encoding with random-init ResNet-18 ==="
    python "${SCRIPT_DIR}/prepare_untrained_resnet.py" \
        --index-dir "${SCRIPT_DIR}/outputs/index" \
        --out       "${H5}" \
        --batch-size 256
else
    echo "[skip] ${H5} already exists"
fi

if [[ ! -f "${CSV}" ]]; then
    echo "=== Probing untrained-resnet embeddings ==="
    python "${SCRIPT_DIR}/predict_combined.py" \
        --embeddings "${H5}" \
        --out        "${CSV}"
else
    echo "[skip] ${CSV} already exists"
fi

echo "Done at $(date)"
