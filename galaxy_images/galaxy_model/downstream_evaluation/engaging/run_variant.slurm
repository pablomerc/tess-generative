#!/bin/bash
#SBATCH --job-name=engaging-variant
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=4:00:00
#SBATCH --partition=mit_normal_gpu
#SBATCH --mem=64G
#SBATCH --output=/orcd/pool/007/pablomer/tess-generative/galaxy_images/galaxy_model/downstream_evaluation/engaging/logs/%x_%j.out
#SBATCH --error=/orcd/pool/007/pablomer/tess-generative/galaxy_images/galaxy_model/downstream_evaluation/engaging/logs/%x_%j.err
#SBATCH --chdir=/orcd/pool/007/pablomer/tess-generative
#SBATCH --open-mode=append

# Per-variant pipeline: prepare embeddings -> probe -> publication plot -> Discord push.
#
# Required env vars (set by run_per_variant.sh via --export):
#   VARIANT       e.g. "base-15NB"
#   CKPT          full path to the .ckpt file
#   MODULE        relative to galaxy_model/  (e.g. "double_train_fm_neighbors.py")
#   MODEL_CLASS   LightningModule class name (e.g. "ConditionalFlowMatchingModule")

set -eo pipefail

: "${VARIANT:?VARIANT not set}"
: "${CKPT:?CKPT not set}"
: "${MODULE:?MODULE not set}"
: "${MODEL_CLASS:?MODEL_CLASS not set}"

SCRIPT_DIR="/home/pablomer/orcd/pool/tess-generative/galaxy_images/galaxy_model/downstream_evaluation/engaging"
REPO_ROOT="/home/pablomer/orcd/pool/tess-generative"

source ~/.bashrc
conda activate torchenv
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export TORCH_BLAS_PREFER_HIPBLASLT=0
export OMP_NUM_THREADS=1
export WANDB_MODE=disabled

cd "${REPO_ROOT}"

mkdir -p "${SCRIPT_DIR}/outputs/embeddings" \
         "${SCRIPT_DIR}/outputs/predictions" \
         "${SCRIPT_DIR}/outputs/plots_publication"

H5="${SCRIPT_DIR}/outputs/embeddings/${VARIANT}.h5"
CSV="${SCRIPT_DIR}/outputs/predictions/predict_${VARIANT}.csv"

echo "=== ${VARIANT} ==="
echo "  ckpt   : ${CKPT}"
echo "  module : ${MODULE}"
echo "  class  : ${MODEL_CLASS}"

if [[ ! -f "${H5}" ]]; then
    python "${SCRIPT_DIR}/prepare_combined.py" \
        --checkpoint  "${CKPT}" \
        --module      "${MODULE}" \
        --model-class "${MODEL_CLASS}" \
        --index-dir   "${SCRIPT_DIR}/outputs/index" \
        --out         "${H5}" \
        --batch-size  256
else
    echo "[skip] ${H5} already exists"
fi

if [[ ! -f "${CSV}" ]]; then
    python "${SCRIPT_DIR}/predict_combined.py" \
        --embeddings "${H5}" \
        --out        "${CSV}"
else
    echo "[skip] ${CSV} already exists"
fi

# Plot is regenerated every run so the latest figure always lands on Discord.
python "${SCRIPT_DIR}/plot_aion_vs_variant.py" --variant "${VARIANT}"

echo "Done at $(date)"
