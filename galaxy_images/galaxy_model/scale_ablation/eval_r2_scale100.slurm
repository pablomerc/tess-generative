#!/bin/bash
#SBATCH --job-name=engaging-scale-100
#SBATCH --partition=mit_normal_gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/home/pablomer/orcd/pool/tess-generative/galaxy_images/galaxy_model/scale_ablation/logs/%x_%j.out
#SBATCH --error=/home/pablomer/orcd/pool/tess-generative/galaxy_images/galaxy_model/scale_ablation/logs/%x_%j.err
#SBATCH --chdir=/orcd/pool/007/pablomer/tess-generative
#SBATCH --open-mode=append
#
# Downstream R^2 for the n=100 rung — the extra arm added below the original 1k/3.16k/10k/31.6k
# ladder. Does in ONE job what run_variant.slurm + eval_meanpool_scale.slurm did for the other
# four arms: flat embeddings -> probe, then the token-mean-pooled twin -> probe.
#
# The meanpool twin is not optional: report_scale_results.py takes the INSTRUMENT column from
# predict_<v>-meanpool.csv (pooled 32-d combined_e2) and only falls back to the flat 128-d with
# a "†" flag. Without the twin the n=100 instrument numbers would sit on a different latent
# than the other four arms and the column would not be comparable.
#
# Idempotent: every stage skips if its output already exists.

set -eo pipefail

E="/home/pablomer/orcd/pool/tess-generative/galaxy_images/galaxy_model/downstream_evaluation/engaging"
ABL="/home/pablomer/orcd/pool/tess-generative/galaxy_images/galaxy_model/scale_ablation"
FROZEN="/orcd/pool/007/pablomer/checkpoints_new/scale_ablation/frozen"

VARIANT="scale-100"
CKPT="${FROZEN}/scale_100/scale-100-75k.ckpt"
MODULE="double_train_fm_neighbors.py"
MODEL_CLASS="ConditionalFlowMatchingModule"

mkdir -p "${ABL}/logs"

WEBHOOK="${DISCORD_WEBHOOK:-https://discord.com/api/webhooks/1500242707522519183/NqqAxuo5n0Ex4X81vMcq7gDOjrP3oAURAVZtgrZT5CramGNfBGPxuEvC2NnxwYBkfs9j}"
_ping() { curl -sS -X POST -H 'Content-Type: application/json' -d "{\"content\": \"$1\"}" "${WEBHOOK}" >/dev/null 2>&1 || true; }
_ping "engaging-scale-100 🚀 start · downstream R² for the n=100 arm (flat + meanpool) · job ${SLURM_JOB_ID:-?}"
trap '_ping "engaging-scale-100 $([ $? -eq 0 ] && echo ✅ || echo ❌) · job ${SLURM_JOB_ID:-?}"' EXIT

source ~/.bashrc
conda activate torchenv
export PYTHONPATH="/home/pablomer/orcd/pool/tess-generative:${PYTHONPATH:-}"
export TORCH_BLAS_PREFER_HIPBLASLT=0
export OMP_NUM_THREADS=1
export WANDB_MODE=disabled

cd /home/pablomer/orcd/pool/tess-generative

test -f "${CKPT}" || { echo "FATAL: frozen checkpoint missing: ${CKPT}"; exit 1; }

H5="${E}/outputs/embeddings/${VARIANT}.h5"
CSV="${E}/outputs/predictions/predict_${VARIANT}.csv"
PH5="${E}/outputs/embeddings/${VARIANT}-meanpool.h5"
PCSV="${E}/outputs/predictions/predict_${VARIANT}-meanpool.csv"

echo "=== ${VARIANT} ==="
echo "  ckpt : ${CKPT}"

if [[ ! -f "${H5}" ]]; then
    python "${E}/prepare_combined.py" \
        --checkpoint  "${CKPT}" \
        --module      "${MODULE}" \
        --model-class "${MODEL_CLASS}" \
        --index-dir   "${E}/outputs/index" \
        --out         "${H5}" \
        --batch-size  256
else
    echo "[skip] ${H5} exists"
fi

if [[ ! -f "${CSV}" ]]; then
    python "${E}/predict_combined.py" --embeddings "${H5}" --out "${CSV}"
else
    echo "[skip] ${CSV} exists"
fi

if [[ ! -f "${PH5}" ]]; then
    python "${E}/build_meanpool_h5.py" --in "${H5}" --out "${PH5}" --tokens 4
else
    echo "[skip] ${PH5} exists"
fi

if [[ ! -f "${PCSV}" ]]; then
    python "${E}/predict_combined.py" --embeddings "${PH5}" --out "${PCSV}"
else
    echo "[skip] ${PCSV} exists"
fi

# Publication plot, same as run_variant.slurm does for every other arm. Non-fatal: the
# R^2 numbers are already on disk by this point and the scale table does not read the plot.
python "${E}/plot_aion_vs_variant.py" --variant "${VARIANT}" || echo "[warn] plot step failed (non-fatal)"

echo "[engaging-scale-100] done at $(date)"
ls -l "${CSV}" "${PCSV}"
