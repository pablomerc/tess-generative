#!/bin/bash
#SBATCH --job-name=jepa-width-ctrl
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --partition=mit_normal_gpu
#SBATCH --mem=32G
#SBATCH --output=/orcd/pool/007/pablomer/tess-generative/galaxy_images/galaxy_model/jepa_ablation/logs/%x_%j.out
#SBATCH --error=/orcd/pool/007/pablomer/tess-generative/galaxy_images/galaxy_model/jepa_ablation/logs/%x_%j.err
#SBATCH --open-mode=append
#
# Width control for the hsc-vit arm: PCA-reduce its 192-d embeddings to 64-d and
# re-run the identical probe, so its R² is comparable to the 64-d arms rather than
# confounded by a 3x wider probe input. Same convention the paper uses for AION
# (aion/hsc_mean_pca64).
#
#   sbatch --export=ALL,ARM=hsc-vit width_control.slurm

set -eo pipefail

REPO_ROOT="/home/pablomer/orcd/pool/tess-generative"
GM_DIR="${REPO_ROOT}/galaxy_images/galaxy_model"
ENG_DIR="${GM_DIR}/downstream_evaluation/engaging"
JEPA_DIR="${GM_DIR}/jepa_ablation"

ARM="${ARM:-hsc-vit}"
NCOMP="${NCOMP:-64}"
VAR="jepa-${ARM}"
VAR_PCA="jepa-${ARM}-pca${NCOMP}"

WEBHOOK="${DISCORD_WEBHOOK:-https://discord.com/api/webhooks/1500242707522519183/NqqAxuo5n0Ex4X81vMcq7gDOjrP3oAURAVZtgrZT5CramGNfBGPxuEvC2NnxwYBkfs9j}"
_ping() { curl -sS -X POST -H 'Content-Type: application/json' -d "{\"content\": \"$1\"}" "${WEBHOOK}" >/dev/null 2>&1 || true; }
_on_exit() { rc=$?; if [[ $rc -ne 0 ]]; then _ping "${VAR_PCA} width control ❌ failed (rc=$rc) job ${SLURM_JOB_ID:-?}"; fi; exit $rc; }
trap _on_exit EXIT

source ~/.bashrc
conda activate torchenv
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1
export WANDB_MODE=disabled
cd "${REPO_ROOT}"

EMB="${ENG_DIR}/outputs/embeddings/${VAR}.h5"
EMB_PCA="${ENG_DIR}/outputs/embeddings/${VAR_PCA}.h5"
CSV_PCA="${ENG_DIR}/outputs/predictions/predict_${VAR_PCA}__aion.csv"

if [[ ! -f "${EMB}" ]]; then
    _ping "${VAR_PCA} ❌ source embeddings missing: ${EMB}"
    exit 1
fi

_ping "${VAR_PCA} 📐 width control started (PCA ${NCOMP}-d) job ${SLURM_JOB_ID:-?}"

python "${JEPA_DIR}/pca_reduce_embeddings.py" --in "${EMB}" --out "${EMB_PCA}" -n "${NCOMP}"

python "${ENG_DIR}/predict_combined.py" \
    --embeddings "${EMB_PCA}" \
    --out        "${CSV_PCA}" \
    --mlp-arch   aion \
    --seed       0

_ping "${VAR_PCA} ✅ width control done — R² at matched ${NCOMP}-d written to $(basename "${CSV_PCA}")"
echo "Done at $(date)"
