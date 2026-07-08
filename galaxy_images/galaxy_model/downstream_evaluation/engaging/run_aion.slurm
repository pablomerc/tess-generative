#!/bin/bash
#SBATCH --job-name=engaging-aion
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=4:00:00
#SBATCH --partition=mit_normal_gpu
#SBATCH --mem=64G
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --open-mode=append

# AION-base downstream eval on the engaging dataset:
#   1. prepare_aion.py     -> outputs/embeddings/aion-base.h5
#   2. predict_combined.py -> outputs/predictions/predict_aion-base.csv
#   3. makeplot.py         -> regenerates compare_*.png + compare_all_with_baselines.png

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
    if [[ $rc -eq 0 ]]; then
        _ping_file "engaging-aion ${SLURM_JOB_ID:-?} ✅ done at $(date '+%F %T') — full plot incl. fresh aion-base" \
                   "${SCRIPT_DIR}/outputs/plots/compare_all_with_baselines.png"
    else
        _ping "engaging-aion ${SLURM_JOB_ID:-?} ❌ failed (rc=$rc) at $(date '+%F %T')"
    fi
}
trap _on_exit EXIT
_ping "engaging-aion ${SLURM_JOB_ID:-?} started at $(date '+%F %T')"

source ~/.bashrc
conda activate torchenv
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export TORCH_BLAS_PREFER_HIPBLASLT=0
export OMP_NUM_THREADS=1

cd "${REPO_ROOT}"

mkdir -p "${SCRIPT_DIR}/outputs/embeddings" \
         "${SCRIPT_DIR}/outputs/predictions" \
         "${SCRIPT_DIR}/outputs/plots"

echo "=== [1/3] Encoding with AION-base ==="
python "${SCRIPT_DIR}/prepare_aion.py" \
    --index-dir "${SCRIPT_DIR}/outputs/index" \
    --out       "${SCRIPT_DIR}/outputs/embeddings/aion-base.h5" \
    --batch-size 16

echo ""
echo "=== [2/3] Probing AION-base latents ==="
python "${SCRIPT_DIR}/predict_combined.py" \
    --embeddings "${SCRIPT_DIR}/outputs/embeddings/aion-base.h5" \
    --out        "${SCRIPT_DIR}/outputs/predictions/predict_aion-base.csv"

echo ""
echo "=== [3/3] Re-plotting with aion-base + all existing checkpoints ==="
python "${SCRIPT_DIR}/makeplot.py" \
    --predictions-dir "${SCRIPT_DIR}/outputs/predictions" \
    --out-dir         "${SCRIPT_DIR}/outputs/plots" \
    --checkpoints     base hier hier-small single-baseline aion-base

echo "Done at $(date)"
