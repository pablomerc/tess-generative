#!/bin/bash
#SBATCH --job-name=jepa-train
#SBATCH --partition=pg_mki_aryeh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=/orcd/pool/007/pablomer/tess-generative/galaxy_images/galaxy_model/jepa_ablation/logs/%x_%j.out
#SBATCH --error=/orcd/pool/007/pablomer/tess-generative/galaxy_images/galaxy_model/jepa_ablation/logs/%x_%j.err
#SBATCH --open-mode=append
#
# JEPA ablation training — ENGAGING (pg_mki_aryeh, 1x H200). See PLAN.md.
#
# Self-chains like the contrastive baseline (afterany successor queued BEFORE
# training; DONE marker stops the chain). 24h on this partition should finish a
# 100k-step run in one link — the chain is insurance.
#
# Checkpoints: canonical path under checkpoints_new/ is a SYMLINK into the 200TB
# mki_aryeh group allocation (POOL is >90% full), same pattern as scale_ablation.
#
#   sbatch --export=ALL,ARM=cf --job-name=jepa-cf train_jepa.slurm
#
# Optional env: ENCODER_POOL (conv1x1), MAX_STEPS (100000), BATCH_SIZE (64),
#               LR (1e-4), MAX_ATTEMPTS (10), DISCORD_WEBHOOK.

set -eo pipefail

REPO_ROOT="/home/pablomer/orcd/pool/tess-generative"
JEPA_DIR="${REPO_ROOT}/galaxy_images/galaxy_model/jepa_ablation"

: "${ARM:?set ARM (hsc-resnet|hsc-vit|cf|cross)}"
ENCODER_POOL="${ENCODER_POOL:-conv1x1}"
MAX_STEPS="${MAX_STEPS:-100000}"
BATCH_SIZE="${BATCH_SIZE:-64}"
LR="${LR:-1e-4}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-10}"

RUN_TAG="jepa-${ARM}"
REAL_DIR="/orcd/data/mki_aryeh/001/pablomer/jepa_ablation/${RUN_TAG}"
CKPT_DIR="/orcd/pool/007/pablomer/checkpoints_new/${RUN_TAG}"
mkdir -p "${REAL_DIR}" "${JEPA_DIR}/logs"
if [[ ! -e "${CKPT_DIR}" ]]; then
    ln -s "${REAL_DIR}" "${CKPT_DIR}"
fi

WEBHOOK="${DISCORD_WEBHOOK:-https://discord.com/api/webhooks/1500242707522519183/NqqAxuo5n0Ex4X81vMcq7gDOjrP3oAURAVZtgrZT5CramGNfBGPxuEvC2NnxwYBkfs9j}"
_ping() { curl -sS -X POST -H 'Content-Type: application/json' -d "{\"content\": \"$1\"}" "${WEBHOOK}" >/dev/null 2>&1 || true; }

# --- Stop the chain if training already finished ---
if [[ -f "${CKPT_DIR}/DONE" ]]; then
    echo "[chain] ${CKPT_DIR}/DONE exists — training complete, nothing to do."
    exit 0
fi

# --- Bound crash loops ---
ATTEMPT=$(cat "${CKPT_DIR}/attempts" 2>/dev/null || echo 0)
ATTEMPT=$((ATTEMPT + 1))
echo "${ATTEMPT}" > "${CKPT_DIR}/attempts"
echo "[chain] attempt ${ATTEMPT}/${MAX_ATTEMPTS} for ${RUN_TAG} (job ${SLURM_JOB_ID:-?})"
if [[ "${ATTEMPT}" -gt "${MAX_ATTEMPTS}" ]]; then
    _ping "${RUN_TAG} ❌ exceeded ${MAX_ATTEMPTS} attempts — giving up (job ${SLURM_JOB_ID:-?})"
    exit 1
fi

# --- Queue the successor now, so a wall-limit kill still continues the run ---
sbatch --dependency=afterany:${SLURM_JOB_ID} \
       --job-name="${SLURM_JOB_NAME}" \
       --export=ALL,ARM=${ARM},ENCODER_POOL=${ENCODER_POOL},MAX_STEPS=${MAX_STEPS},BATCH_SIZE=${BATCH_SIZE},LR=${LR},MAX_ATTEMPTS=${MAX_ATTEMPTS} \
       "${JEPA_DIR}/train_jepa.slurm" || echo "[chain] WARNING: could not queue successor"

source ~/.bashrc
conda activate torchenv
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
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
    _ping "${RUN_TAG} 🚀 training started · pool=${ENCODER_POOL} · target ${MAX_STEPS} steps · job ${SLURM_JOB_ID:-?} · host ${SLURMD_NODENAME:-?}"
fi

cd "${REPO_ROOT}"
python "${JEPA_DIR}/train_jepa.py" \
    --arm           "${ARM}" \
    --encoder-pool  "${ENCODER_POOL}" \
    --max-steps     "${MAX_STEPS}" \
    --batch-size    "${BATCH_SIZE}" \
    --lr            "${LR}" \
    --ckpt-dir      "${CKPT_DIR}" \
    --run-name      "${RUN_TAG}"

# --- Reached here => python exited cleanly. If DONE, launch the downstream eval. ---
if [[ -f "${CKPT_DIR}/DONE" ]]; then
    _ping "${RUN_TAG} ✅ training finished (${MAX_STEPS} steps) — launching downstream eval"
    sbatch --job-name="jepa-eval-${ARM}" --export=ALL,ARM=${ARM} "${JEPA_DIR}/eval_jepa.slurm" \
        || _ping "${RUN_TAG} ⚠️ failed to submit eval job"
fi
