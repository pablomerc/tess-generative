#!/bin/bash
#SBATCH --job-name=engaging-matched-pool-plot
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=1:00:00
#SBATCH --partition=mit_normal_gpu
#SBATCH --mem=16G
#SBATCH --output=/orcd/pool/007/pablomer/tess-generative/galaxy_images/galaxy_model/downstream_evaluation/engaging/logs/%x_%j.out
#SBATCH --error=/orcd/pool/007/pablomer/tess-generative/galaxy_images/galaxy_model/downstream_evaluation/engaging/logs/%x_%j.err
#SBATCH --chdir=/orcd/pool/007/pablomer/tess-generative
#SBATCH --open-mode=append

# Stage 2 of the matched-resnet experiment + the hybrid (phys-flat /
# instr-pooled) plot. Depends on engaging-untr-matched having produced
# untrained-resnet-matched.h5 and predict_untrained-resnet-matched.csv.

set -eo pipefail

SCRIPT_DIR="/home/pablomer/orcd/pool/tess-generative/galaxy_images/galaxy_model/downstream_evaluation/engaging"
REPO_ROOT="/home/pablomer/orcd/pool/tess-generative"

DISCORD_WEBHOOK="${DISCORD_WEBHOOK:-https://discord.com/api/webhooks/1500242707522519183/NqqAxuo5n0Ex4X81vMcq7gDOjrP3oAURAVZtgrZT5CramGNfBGPxuEvC2NnxwYBkfs9j}"
_ping() { curl -sS -X POST -H 'Content-Type: application/json' -d "{\"content\": \"$1\"}" "${DISCORD_WEBHOOK}" >/dev/null || true; }
_ping_file() {
    if [[ -f "$2" ]]; then
        curl -sS -F "payload_json={\"content\":\"$1\"}" -F "file=@$2" "${DISCORD_WEBHOOK}" >/dev/null || true
    else
        _ping "$1 (file missing: $2)"
    fi
}
_on_exit() {
    rc=$?
    if [[ $rc -ne 0 ]]; then
        _ping "engaging-matched-pool-plot ${SLURM_JOB_ID:-?} ❌ failed (rc=$rc) at $(date '+%F %T')"
    fi
}
trap _on_exit EXIT
_ping "engaging-matched-pool-plot ${SLURM_JOB_ID:-?} started at $(date '+%F %T')"

source ~/.bashrc
conda activate torchenv
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export TORCH_BLAS_PREFER_HIPBLASLT=0
export OMP_NUM_THREADS=1
export WANDB_MODE=disabled

cd "${REPO_ROOT}"

H5_IN="${SCRIPT_DIR}/outputs/embeddings/untrained-resnet-matched.h5"
H5_OUT="${SCRIPT_DIR}/outputs/embeddings/untrained-resnet-matched-meanpool.h5"
CSV="${SCRIPT_DIR}/outputs/predictions/predict_untrained-resnet-matched-meanpool.csv"

if [[ ! -f "${H5_IN}" ]]; then
    echo "ERROR: ${H5_IN} not found — upstream job must have failed."
    exit 2
fi

if [[ ! -f "${H5_OUT}" ]]; then
    python "${SCRIPT_DIR}/build_meanpool_h5.py" \
        --in  "${H5_IN}" \
        --out "${H5_OUT}"
else
    echo "[skip] ${H5_OUT} already exists"
fi

if [[ ! -f "${CSV}" ]]; then
    python "${SCRIPT_DIR}/predict_combined.py" \
        --embeddings "${H5_OUT}" \
        --out        "${CSV}"
else
    echo "[skip] ${CSV} already exists"
fi

# Hybrid plot (also pushes to Discord on success).
python "${SCRIPT_DIR}/plot_phys_flat_inst_pooled.py"

_ping_file "engaging-matched-pool-plot ${SLURM_JOB_ID:-?} ✅ done — phys-flat + instr-pooled plot" \
           "${SCRIPT_DIR}/outputs/plots_publication/phys_flat_inst_pooled_all.png"

echo "Done at $(date)"
