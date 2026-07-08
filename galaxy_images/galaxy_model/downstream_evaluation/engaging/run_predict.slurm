#!/bin/bash
#SBATCH --job-name=engaging-predict
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=4:00:00
#SBATCH --partition=mit_normal_gpu
#SBATCH --mem=64G
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --open-mode=append

# Trains MLP probes for every checkpoint H5 produced by run_prepare.slurm,
# then makes the comparison plots.

set -eo pipefail

SCRIPT_DIR="/home/pablomer/orcd/pool/tess-generative/galaxy_images/galaxy_model/downstream_evaluation/engaging"
REPO_ROOT="/home/pablomer/orcd/pool/tess-generative"

DISCORD_WEBHOOK="${DISCORD_WEBHOOK:-https://discord.com/api/webhooks/1500242707522519183/NqqAxuo5n0Ex4X81vMcq7gDOjrP3oAURAVZtgrZT5CramGNfBGPxuEvC2NnxwYBkfs9j}"
_ping() { curl -sS -X POST -H 'Content-Type: application/json' -d "{\"content\": \"$1\"}" "${DISCORD_WEBHOOK}" >/dev/null || true; }
_ping_file() {
    # $1 = message, $2 = file path
    if [[ -f "$2" ]]; then
        curl -sS -F "payload_json={\"content\":\"$1\"}" -F "file=@$2" "${DISCORD_WEBHOOK}" >/dev/null || true
    else
        _ping "$1 (file missing: $2)"
    fi
}
_on_exit() {
    rc=$?
    if [[ $rc -eq 0 ]]; then
        _ping_file "engaging-predict ${SLURM_JOB_ID:-?} ✅ done at $(date '+%F %T') — full plot (combined latents + OURS-FEB + AION)" \
                   "${SCRIPT_DIR}/outputs/plots/compare_all_with_baselines.png"
    else
        _ping "engaging-predict ${SLURM_JOB_ID:-?} ❌ failed (rc=$rc) at $(date '+%F %T')"
    fi
}
trap _on_exit EXIT
_ping "engaging-predict ${SLURM_JOB_ID:-?} started at $(date '+%F %T')"

source ~/.bashrc
conda activate torchenv
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export TORCH_BLAS_PREFER_HIPBLASLT=0
export OMP_NUM_THREADS=1

cd "${REPO_ROOT}"

mkdir -p "${SCRIPT_DIR}/outputs/predictions" "${SCRIPT_DIR}/outputs/plots"

shopt -s nullglob
for h5 in "${SCRIPT_DIR}"/outputs/embeddings/*.h5; do
    name="$(basename "${h5}" .h5)"
    echo "=== Predicting for ${name} ==="
    python "${SCRIPT_DIR}/predict_combined.py" \
        --embeddings "${h5}" \
        --out        "${SCRIPT_DIR}/outputs/predictions/predict_${name}.csv"
done

echo "=== Plotting ==="
python "${SCRIPT_DIR}/makeplot.py" \
    --predictions-dir "${SCRIPT_DIR}/outputs/predictions" \
    --out-dir         "${SCRIPT_DIR}/outputs/plots"

echo "Done at $(date)"
