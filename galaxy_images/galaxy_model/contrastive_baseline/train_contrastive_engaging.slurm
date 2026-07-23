#!/bin/bash
#SBATCH --job-name=contr-train
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --time=06:00:00
#SBATCH --partition=mit_normal_gpu
#SBATCH --mem=64G
#SBATCH --output=/orcd/pool/007/pablomer/tess-generative/galaxy_images/galaxy_model/contrastive_baseline/logs/%x_%j.out
#SBATCH --error=/orcd/pool/007/pablomer/tess-generative/galaxy_images/galaxy_model/contrastive_baseline/logs/%x_%j.err
#SBATCH --open-mode=append
#
# Self-chaining contrastive trainer for Engaging.
#
# mit_normal_gpu caps wall time at ~6h, which is shorter than a 100k-step run.
# To span that, THIS job queues a successor (afterany) BEFORE it starts training;
# if the wall limit (or a node failure) kills this job, the successor resumes from
# the rolling `last.ckpt`. The python entry writes a DONE marker when it reaches
# --max-steps, at which point (a) the successor exits immediately at the DONE
# check below, and (b) this job launches the downstream eval job.
#
# Parameterize with:
#   sbatch --export=ALL,RUN_TAG=spatial-avg,ENCODER_POOL=avg,RANDOM_NEIGHBORS=0,MAX_STEPS=100000 \
#          train_contrastive_engaging.slurm

set -eo pipefail

REPO_ROOT="/home/pablomer/orcd/pool/tess-generative"
CB_DIR="${REPO_ROOT}/galaxy_images/galaxy_model/contrastive_baseline"
ENG_DIR="${REPO_ROOT}/galaxy_images/galaxy_model/downstream_evaluation/engaging"

: "${RUN_TAG:?set RUN_TAG (e.g. spatial-avg)}"
: "${ENCODER_POOL:?set ENCODER_POOL (avg|conv1x1)}"
: "${RANDOM_NEIGHBORS:?set RANDOM_NEIGHBORS (0|1)}"
MAX_STEPS="${MAX_STEPS:-100000}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-15}"
CKPT_DIR="/orcd/pool/007/pablomer/checkpoints_new/contrastive-${RUN_TAG}"
mkdir -p "${CKPT_DIR}" "${CB_DIR}/logs"

WEBHOOK="${DISCORD_WEBHOOK:-https://discord.com/api/webhooks/1500242707522519183/NqqAxuo5n0Ex4X81vMcq7gDOjrP3oAURAVZtgrZT5CramGNfBGPxuEvC2NnxwYBkfs9j}"
_ping() { curl -sS -X POST -H 'Content-Type: application/json' -d "{\"content\": \"$1\"}" "${WEBHOOK}" >/dev/null 2>&1 || true; }

# --- Stop the chain if training already finished (checked BEFORE anything else) ---
if [[ -f "${CKPT_DIR}/DONE" ]]; then
    echo "[chain] ${CKPT_DIR}/DONE exists — training complete, nothing to do."
    exit 0
fi

# --- Bound crash loops ---
ATTEMPT=$(cat "${CKPT_DIR}/attempts" 2>/dev/null || echo 0)
ATTEMPT=$((ATTEMPT + 1))
echo "${ATTEMPT}" > "${CKPT_DIR}/attempts"
echo "[chain] attempt ${ATTEMPT}/${MAX_ATTEMPTS} for contrastive-${RUN_TAG} (job ${SLURM_JOB_ID:-?})"
if [[ "${ATTEMPT}" -gt "${MAX_ATTEMPTS}" ]]; then
    _ping "contr-${RUN_TAG} ❌ exceeded ${MAX_ATTEMPTS} attempts — giving up (job ${SLURM_JOB_ID:-?})"
    exit 1
fi

# --- Queue the successor now, so a wall-limit kill still continues the run ---
sbatch --dependency=afterany:${SLURM_JOB_ID} \
       --export=ALL,RUN_TAG=${RUN_TAG},ENCODER_POOL=${ENCODER_POOL},RANDOM_NEIGHBORS=${RANDOM_NEIGHBORS},MAX_STEPS=${MAX_STEPS},MAX_ATTEMPTS=${MAX_ATTEMPTS} \
       "${CB_DIR}/train_contrastive_engaging.slurm" || echo "[chain] WARNING: could not queue successor"

source ~/.bashrc
conda activate torchenv
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export TORCH_BLAS_PREFER_HIPBLASLT=0     # harmless no-op on NVIDIA
# Single-threaded numpy/BLAS per DataLoader worker (avoid core oversubscription).
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# WandB: online if authenticated (env key OR a ~/.netrc wandb login), else offline.
if [[ -n "${WANDB_API_KEY:-}" ]] || grep -q "api.wandb.ai" ~/.netrc 2>/dev/null; then
    export WANDB_MODE=online
else
    export WANDB_MODE="${WANDB_MODE:-offline}"
fi
export WANDB_DIR="${CKPT_DIR}"
echo "[wandb] WANDB_MODE=${WANDB_MODE}  WANDB_DIR=${WANDB_DIR}"

if [[ "${ATTEMPT}" -eq 1 ]]; then
    _ping "contr-${RUN_TAG} 🚀 training started · pool=${ENCODER_POOL} random_neighbors=${RANDOM_NEIGHBORS} · target ${MAX_STEPS} steps · job ${SLURM_JOB_ID:-?}"
fi

cd "${REPO_ROOT}"
python "${CB_DIR}/train_contrastive_engaging.py" \
    --encoder-pool    "${ENCODER_POOL}" \
    --random-neighbors "${RANDOM_NEIGHBORS}" \
    --max-steps       "${MAX_STEPS}" \
    --ckpt-dir        "${CKPT_DIR}" \
    --run-name        "contrastive-${RUN_TAG}"

# --- Reached here => python exited cleanly. If DONE, launch the downstream eval. ---
if [[ -f "${CKPT_DIR}/DONE" ]]; then
    _ping "contr-${RUN_TAG} ✅ training finished (${MAX_STEPS} steps) — launching downstream eval"
    sbatch --export=ALL,RUN_TAG=${RUN_TAG} "${CB_DIR}/eval_contrastive_engaging.slurm" \
        || _ping "contr-${RUN_TAG} ⚠️ failed to submit eval job"
fi
