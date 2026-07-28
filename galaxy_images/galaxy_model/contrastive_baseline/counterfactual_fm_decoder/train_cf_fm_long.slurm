#!/bin/bash
#SBATCH --job-name=cf-fm-long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --partition=pg_mki_aryeh
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --open-mode=append
#
# Aryeh trainer: 100k steps in ONE 2-day job, no auto-resubmit. Defaults to
# tokens/xattn (plan B).
#
# Preferred over train_cf_fm_tokens.slurm on aryeh: that script self-resubmits with
# `sbatch $0`, so its own `#SBATCH --partition=mit_normal_gpu` wins on every segment
# after the first — passing `-p pg_mki_aryeh` on the command line only moves segment 1.
# One long job also avoids the mit_normal_gpu 2-GPU QOS cap entirely.
#
# Override: COND_STYLE=meanpool INJECTION=adagn RUN_DIR=... sbatch train_cf_fm_long.slurm

set -eo pipefail

REPO=/orcd/pool/007/pablomer/tess-generative
TORCHENV_PY=/orcd/home/002/pablomer/miniforge3/envs/torchenv/bin/python
# DATA_H5 and RUN_DIR must be changed TOGETHER when switching frozen encoders — the
# H5 carries the embeddings a given decoder was trained on, and both heads emit 64-D so
# a mismatch cannot be caught by a shape check. TAG keeps run dir / W&B name in step.
: "${DATA_H5:=/orcd/pool/007/pablomer/outputs/counterfactual_fm/cf_embeddings_all.h5}"
: "${COND_STYLE:=tokens}"
: "${INJECTION:=xattn}"
: "${TAG:=long}"
: "${RUN_DIR:=/orcd/pool/007/pablomer/outputs/counterfactual_fm/runs/cf-fm-${COND_STYLE}-${INJECTION}-${TAG}}"
: "${MAX_STEPS:=100000}"
: "${BATCH:=128}"
# Extra budget snapshots, e.g. 75000 to compare at fm-control's own 75k training budget
# rather than at whichever step happened to minimise val/mse.
: "${SAVE_AT_STEPS:=}"
: "${WANDB_NAME:=cf-fm-${COND_STYLE}-${INJECTION}-${TAG}}"
export WANDB_API_KEY="${WANDB_API_KEY:-}"
WEBHOOK="${DISCORD_WEBHOOK:-https://discord.com/api/webhooks/1500242707522519183/NqqAxuo5n0Ex4X81vMcq7gDOjrP3oAURAVZtgrZT5CramGNfBGPxuEvC2NnxwYBkfs9j}"

_ping() { curl -sS -X POST -H 'Content-Type: application/json' -d "{\"content\": \"$1\"}" "${WEBHOOK}" >/dev/null 2>&1 || true; }

mkdir -p "$RUN_DIR/checkpoints"

source ~/.bashrc
conda activate torchenv || true
export PYTHONPATH="$REPO:$PYTHONPATH"
cd "$REPO"

# Online if an API key is exported or ~/.netrc carries the wandb entry; otherwise
# offline, so a missing credential degrades instead of blocking the run.
if [[ -n "${WANDB_API_KEY:-}" ]] || grep -q "api.wandb.ai" ~/.netrc 2>/dev/null; then
    export WANDB_MODE=online
else
    export WANDB_MODE="${WANDB_MODE:-offline}"
fi
echo "[wandb] WANDB_MODE=${WANDB_MODE} | project=galaxy-counterfactual-fm-decoder | name=${WANDB_NAME}"

_ping "cf-fm-long 🚀 start · ${COND_STYLE}/${INJECTION} · job ${SLURM_JOB_ID:-?} · host ${SLURMD_NODENAME:-?} · W&B \`galaxy-counterfactual-fm-decoder\`/\`${WANDB_NAME}\` (${WANDB_MODE})"
cleanup() {
  local rc=$?
  if [[ -f "$RUN_DIR/DONE" ]]; then
    _ping "cf-fm-long ✅ reached max-steps · \`${RUN_DIR}\` · job ${SLURM_JOB_ID:-?}"
  elif [[ ${rc} -eq 0 ]]; then
    _ping "cf-fm-long ⏹ exited 0 without DONE · job ${SLURM_JOB_ID:-?}"
  else
    _ping "cf-fm-long ❌ failed (exit ${rc}) · job ${SLURM_JOB_ID:-?} · resume from \`last.ckpt\`"
  fi
}
trap cleanup EXIT

RESUME_ARG=""
if [ -f "$RUN_DIR/checkpoints/last.ckpt" ]; then
    RESUME_ARG="--resume-from $RUN_DIR/checkpoints/last.ckpt"
    echo "[resume] resuming from $RUN_DIR/checkpoints/last.ckpt"
fi

srun "$TORCHENV_PY" -u galaxy_images/galaxy_model/contrastive_baseline/counterfactual_fm_decoder/train_cf_fm.py \
    --data-h5 "$DATA_H5" \
    --cond-style "$COND_STYLE" \
    --injection "$INJECTION" \
    --run-dir "$RUN_DIR" \
    --wandb-name "$WANDB_NAME" \
    --max-steps "$MAX_STEPS" \
    --batch-size "$BATCH" \
    --val-ratio 0.05 \
    --monitor val/mse \
    --save-at-steps "$SAVE_AT_STEPS" \
    --devices 1 \
    $RESUME_ARG

echo "Long train finished at $(date)"
