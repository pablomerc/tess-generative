#!/bin/bash
#SBATCH --job-name=cf-fm-tokens
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --partition=mit_normal_gpu
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --open-mode=append
#
# Train plan-B: tokens + xattn CF decoder, with 6 h auto-resubmit.
# Submit ONCE: sbatch train_cf_fm_tokens.slurm

set -eo pipefail

# ---- EDIT THESE FOR YOUR CLUSTER ----
REPO=/orcd/pool/007/pablomer/tess-generative
TORCHENV_PY=/orcd/home/002/pablomer/miniforge3/envs/torchenv/bin/python
DATA_H5=/orcd/pool/007/pablomer/outputs/counterfactual_fm/cf_embeddings_all.h5
RUN_DIR=/orcd/pool/007/pablomer/outputs/counterfactual_fm/runs/cf-fm-tokens-xattn
COND_STYLE=tokens
INJECTION=xattn
MAX_STEPS=100000
BATCH=128
MAX_RESUBMITS=30
export WANDB_API_KEY="${WANDB_API_KEY:-}"
WEBHOOK="${DISCORD_WEBHOOK:-https://discord.com/api/webhooks/1500242707522519183/NqqAxuo5n0Ex4X81vMcq7gDOjrP3oAURAVZtgrZT5CramGNfBGPxuEvC2NnxwYBkfs9j}"
# -------------------------------------

_ping() { curl -sS -X POST -H 'Content-Type: application/json' -d "{\"content\": \"$1\"}" "${WEBHOOK}" >/dev/null 2>&1 || true; }

mkdir -p "$RUN_DIR/checkpoints"

if [ -f "$RUN_DIR/DONE" ]; then
    echo "[resubmit] $RUN_DIR/DONE present — training complete, exiting."
    _ping "cf-fm-tokens ✅ already DONE · \`${RUN_DIR}\`"
    exit 0
fi

COUNT_FILE="$RUN_DIR/.resubmit_count"
count=$(cat "$COUNT_FILE" 2>/dev/null || echo 0)
if [ "$count" -ge "$MAX_RESUBMITS" ]; then
    echo "[resubmit] hit MAX_RESUBMITS=$MAX_RESUBMITS — stopping chain."
    _ping "cf-fm-tokens ❌ hit MAX_RESUBMITS=${MAX_RESUBMITS} · job ${SLURM_JOB_ID:-?}"
    exit 1
fi
echo $((count + 1)) > "$COUNT_FILE"

sbatch --dependency=afterany:"$SLURM_JOB_ID" "$0" || echo "[resubmit] could not queue successor"

_ping "cf-fm-tokens 🚀 start · segment $((count + 1)) · job ${SLURM_JOB_ID:-?} · host ${SLURMD_NODENAME:-?} · W&B \`galaxy-counterfactual-fm-decoder\` / \`cf-fm-tokens-xattn\`"
cleanup() {
  local rc=$?
  if [[ -f "$RUN_DIR/DONE" ]]; then
    _ping "cf-fm-tokens ✅ reached max-steps · \`${RUN_DIR}\` · job ${SLURM_JOB_ID:-?}"
  elif [[ ${rc} -eq 0 ]]; then
    _ping "cf-fm-tokens ⏸ segment ok (will resume) · job ${SLURM_JOB_ID:-?}"
  else
    _ping "cf-fm-tokens ❌ failed (exit ${rc}) · job ${SLURM_JOB_ID:-?}"
  fi
}
trap cleanup EXIT

source ~/.bashrc
conda activate torchenv || true
export PYTHONPATH="$REPO:$PYTHONPATH"
cd "$REPO"

# WandB online if key or netrc present
if [[ -n "${WANDB_API_KEY:-}" ]] || grep -q "api.wandb.ai" ~/.netrc 2>/dev/null; then
    export WANDB_MODE=online
else
    export WANDB_MODE="${WANDB_MODE:-offline}"
fi
echo "[wandb] WANDB_MODE=${WANDB_MODE}"

RESUME_ARG=""
if [ -f "$RUN_DIR/checkpoints/last.ckpt" ]; then
    RESUME_ARG="--resume-from $RUN_DIR/checkpoints/last.ckpt"
    echo "[resume] resuming from $RUN_DIR/checkpoints/last.ckpt"
fi

echo "PYTHON=$TORCHENV_PY | COND_STYLE=$COND_STYLE | INJECTION=$INJECTION"
"$TORCHENV_PY" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

srun "$TORCHENV_PY" -u galaxy_images/galaxy_model/contrastive_baseline/counterfactual_fm_decoder/train_cf_fm.py \
    --data-h5 "$DATA_H5" \
    --cond-style "$COND_STYLE" \
    --injection "$INJECTION" \
    --run-dir "$RUN_DIR" \
    --max-steps "$MAX_STEPS" \
    --batch-size "$BATCH" \
    --val-ratio 0.05 \
    --devices 1 \
    $RESUME_ARG

echo "Training segment finished at $(date)"
