#!/bin/bash
#SBATCH --job-name=engaging-three-probes
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=2:00:00
#SBATCH --partition=mit_normal_gpu
#SBATCH --mem=16G
#SBATCH --output=/orcd/pool/007/pablomer/tess-generative/galaxy_images/galaxy_model/downstream_evaluation/engaging/logs/%x_%j.out
#SBATCH --error=/orcd/pool/007/pablomer/tess-generative/galaxy_images/galaxy_model/downstream_evaluation/engaging/logs/%x_%j.err
#SBATCH --chdir=/orcd/pool/007/pablomer/tess-generative
#SBATCH --open-mode=append

# Retrains downstream probes under the multi-target (one MLP per task family)
# setup, for every (encoder variant × MLP arch) combination, then renders the
# unified plot and pushes plot + convergence summary to Discord per arch.

set -eo pipefail

SCRIPT_DIR="/home/pablomer/orcd/pool/tess-generative/galaxy_images/galaxy_model/downstream_evaluation/engaging"
REPO_ROOT="/home/pablomer/orcd/pool/tess-generative"

DISCORD_WEBHOOK="${DISCORD_WEBHOOK:-https://discord.com/api/webhooks/1500242707522519183/NqqAxuo5n0Ex4X81vMcq7gDOjrP3oAURAVZtgrZT5CramGNfBGPxuEvC2NnxwYBkfs9j}"
_ping() { curl -sS -X POST -H 'Content-Type: application/json' -d "{\"content\": \"$1\"}" "${DISCORD_WEBHOOK}" >/dev/null || true; }
_on_exit() {
    rc=$?
    if [[ $rc -ne 0 ]]; then
        _ping "engaging-three-probes ${SLURM_JOB_ID:-?} ❌ failed (rc=$rc) at $(date '+%F %T')"
    fi
}
trap _on_exit EXIT
_ping "engaging-three-probes ${SLURM_JOB_ID:-?} started at $(date '+%F %T')  (2 archs × 4 variants = 8 probe runs + 2 plots)"

source ~/.bashrc
conda activate torchenv
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export TORCH_BLAS_PREFER_HIPBLASLT=0
export OMP_NUM_THREADS=1
export WANDB_MODE=disabled

cd "${REPO_ROOT}"

VARIANTS=("base" "base-meanpool" "untrained-resnet-matched" "aion-base")
ARCHS=("aion" "aion_reg")

mkdir -p "${SCRIPT_DIR}/outputs/predictions" "${SCRIPT_DIR}/outputs/plots_publication"

for arch in "${ARCHS[@]}"; do
    for v in "${VARIANTS[@]}"; do
        H5="${SCRIPT_DIR}/outputs/embeddings/${v}.h5"
        CSV="${SCRIPT_DIR}/outputs/predictions/predict_${v}__${arch}.csv"
        if [[ ! -f "${H5}" ]]; then
            echo "[skip] ${H5} missing — variant ${v} not encoded"
            continue
        fi
        if [[ -f "${CSV}" ]]; then
            echo "[skip] ${CSV} already exists"
            continue
        fi
        echo "=== arch=${arch}  variant=${v} ==="
        python "${SCRIPT_DIR}/predict_combined.py" \
            --embeddings "${H5}" \
            --out        "${CSV}" \
            --mlp-arch   "${arch}"
    done

    echo "=== Plotting arch=${arch} ==="
    python "${SCRIPT_DIR}/plot_unified_three_probes.py" --mlp-arch "${arch}"
done

_ping "engaging-three-probes ${SLURM_JOB_ID:-?} ✅ all done at $(date '+%F %T')"

echo "Done at $(date)"
