#!/bin/bash
#SBATCH --job-name=cf-recon-mse
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --partition=mit_normal_gpu
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --open-mode=append
#
# Matched recon MSE: paper base + fm-control + CF decoder checkpoint(s).

set -eo pipefail

REPO=/orcd/pool/007/pablomer/tess-generative
TORCHENV_PY=/orcd/home/002/pablomer/miniforge3/envs/torchenv/bin/python
CONFIG="$REPO/galaxy_images/galaxy_model/diffusion_ablation/configs/neighbors_fm_control.json"
CF_H5=/orcd/pool/007/pablomer/outputs/counterfactual_fm/cf_embeddings_all.h5
RUNS=/orcd/pool/007/pablomer/outputs/counterfactual_fm/runs
# Candidate run dirs per arm; newest best-*.ckpt across them wins. Both the "-long"
# aryeh dirs and the 6h-chain dirs are listed so the eval finds the checkpoint whichever
# launcher produced it — silently emitting no CF row is the worst outcome here.
CF_TOKENS_DIRS=("$RUNS/cf-fm-tokens-xattn-long/checkpoints" "$RUNS/cf-fm-tokens-xattn/checkpoints")
CF_MEANPOOL_DIRS=("$RUNS/cf-fm-meanpool-adagn-long/checkpoints" "$RUNS/cf-fm-meanpool-adagn/checkpoints")
OUT=/orcd/pool/007/pablomer/outputs/counterfactual_fm/results/recon_mse.csv
N=256
NUM_STEPS=250
WEBHOOK="${DISCORD_WEBHOOK:-https://discord.com/api/webhooks/1500242707522519183/NqqAxuo5n0Ex4X81vMcq7gDOjrP3oAURAVZtgrZT5CramGNfBGPxuEvC2NnxwYBkfs9j}"

_ping() { curl -sS -X POST -H 'Content-Type: application/json' -d "{\"content\": \"$1\"}" "${WEBHOOK}" >/dev/null 2>&1 || true; }
_ping "cf-recon-mse 🚀 start · job ${SLURM_JOB_ID:-?}"
cleanup() {
  local rc=$?
  if [[ ${rc} -eq 0 ]]; then
    _ping "cf-recon-mse ✅ done · \`${OUT}\` · job ${SLURM_JOB_ID:-?}"
  else
    _ping "cf-recon-mse ❌ failed (exit ${rc}) · job ${SLURM_JOB_ID:-?}"
  fi
}
trap cleanup EXIT

source ~/.bashrc
conda activate torchenv || true
export PYTHONPATH="$REPO:$PYTHONPATH"
cd "$REPO"

mkdir -p "$(dirname "$OUT")"

# Resolve the newest best-*.ckpt across all candidate dirs for an arm.
resolve_best() {
    local dir
    for dir in "$@"; do
        [ -d "$dir" ] || continue
        ls -t "$dir"/best-*.ckpt 2>/dev/null | head -n1
    done | head -n1
}

CF_ARGS=()
TOK=$(resolve_best "${CF_TOKENS_DIRS[@]}")
if [ -n "$TOK" ]; then
    CF_ARGS+=(--cf-checkpoint "cf-fm-tokens:$TOK")
    echo "[eval] tokens ckpt=$TOK"
else
    echo "[eval] WARNING: no tokens best-*.ckpt in ${CF_TOKENS_DIRS[*]} — CF tokens row will be MISSING"
fi
MP=$(resolve_best "${CF_MEANPOOL_DIRS[@]}")
if [ -n "$MP" ]; then
    CF_ARGS+=(--cf-checkpoint "cf-fm-meanpool:$MP")
    echo "[eval] meanpool ckpt=$MP"
fi

if [ ${#CF_ARGS[@]} -gt 0 ]; then
    CF_ARGS+=(--cf-h5 "$CF_H5")
fi

srun "$TORCHENV_PY" -u galaxy_images/galaxy_model/contrastive_baseline/counterfactual_fm_decoder/eval_cf_recon_mse.py \
    --config "$CONFIG" \
    --n "$N" \
    --num-steps "$NUM_STEPS" \
    --out "$OUT" \
    "${CF_ARGS[@]}"

echo "Eval done at $(date) -> $OUT"
