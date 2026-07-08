#!/bin/bash
#SBATCH --job-name=engaging-base-meanpool
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=2:00:00
#SBATCH --partition=mit_normal_gpu
#SBATCH --mem=32G
#SBATCH --output=/orcd/pool/007/pablomer/tess-generative/galaxy_images/galaxy_model/downstream_evaluation/engaging/logs/%x_%j.out
#SBATCH --error=/orcd/pool/007/pablomer/tess-generative/galaxy_images/galaxy_model/downstream_evaluation/engaging/logs/%x_%j.err
#SBATCH --chdir=/orcd/pool/007/pablomer/tess-generative
#SBATCH --open-mode=append

# Probe + plot for the token-mean-pooled `base` variant. Embeddings are
# constructed offline by build_meanpool_h5.py; this script only trains the
# MLP probes and renders the publication plot.

set -eo pipefail

VARIANT="base-meanpool"
SCRIPT_DIR="/home/pablomer/orcd/pool/tess-generative/galaxy_images/galaxy_model/downstream_evaluation/engaging"
REPO_ROOT="/home/pablomer/orcd/pool/tess-generative"

DISCORD_WEBHOOK="${DISCORD_WEBHOOK:-https://discord.com/api/webhooks/1500242707522519183/NqqAxuo5n0Ex4X81vMcq7gDOjrP3oAURAVZtgrZT5CramGNfBGPxuEvC2NnxwYBkfs9j}"
_ping() { curl -sS -X POST -H 'Content-Type: application/json' -d "{\"content\": \"$1\"}" "${DISCORD_WEBHOOK}" >/dev/null || true; }
_on_exit() {
    rc=$?
    if [[ $rc -ne 0 ]]; then
        _ping "engaging-${VARIANT} ${SLURM_JOB_ID:-?} ❌ failed (rc=$rc) at $(date '+%F %T')"
    fi
}
trap _on_exit EXIT
_ping "engaging-${VARIANT} ${SLURM_JOB_ID:-?} started at $(date '+%F %T')"

source ~/.bashrc
conda activate torchenv
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export TORCH_BLAS_PREFER_HIPBLASLT=0
export OMP_NUM_THREADS=1
export WANDB_MODE=disabled

cd "${REPO_ROOT}"

H5="${SCRIPT_DIR}/outputs/embeddings/${VARIANT}.h5"
CSV="${SCRIPT_DIR}/outputs/predictions/predict_${VARIANT}.csv"

if [[ ! -f "${H5}" ]]; then
    echo "ERROR: ${H5} not found — run build_meanpool_h5.py first."
    exit 2
fi

mkdir -p "${SCRIPT_DIR}/outputs/predictions" "${SCRIPT_DIR}/outputs/plots_publication"

echo "=== ${VARIANT} ==="
echo "  h5  : ${H5}"
echo "  csv : ${CSV}"

python "${SCRIPT_DIR}/predict_combined.py" \
    --embeddings "${H5}" \
    --out        "${CSV}"

# plot_aion_vs_variant.py pushes the PNG to Discord on success.
python "${SCRIPT_DIR}/plot_aion_vs_variant.py" --variant "${VARIANT}"

echo "Done at $(date)"
