#!/bin/bash
#SBATCH --job-name=cf-recon-mse
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --partition=pg_mki_aryeh
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --open-mode=append
#
# Matched recon MSE: paper base + fm-control + CF decoder checkpoint(s).
#
# Runs on aryeh (H200) rather than mit_normal_gpu: no 2-GPU QOS queue, and the 2.2 GB
# checkpoint loads plus 250-step ODE sampling both benefit. All models are loaded once
# and INTERLEAVED per chunk, so a complete three-way table is reported to Discord after
# the first FIRST_CHUNK galaxies and at every REPORT_AT milestone thereafter.

set -eo pipefail

REPO=/orcd/pool/007/pablomer/tess-generative
TORCHENV_PY=/orcd/home/002/pablomer/miniforge3/envs/torchenv/bin/python
CONFIG="$REPO/galaxy_images/galaxy_model/diffusion_ablation/configs/neighbors_fm_control.json"
: "${CF_H5:=/orcd/pool/007/pablomer/outputs/counterfactual_fm/cf_embeddings_all.h5}"
RUNS=/orcd/pool/007/pablomer/outputs/counterfactual_fm/runs
# Candidate run dirs per arm; newest best-*.ckpt across them wins. Both the "-long"
# aryeh dirs and the 6h-chain dirs are listed so the eval finds the checkpoint whichever
# launcher produced it — silently emitting no CF row is the worst outcome here.
CF_TOKENS_DIRS=("$RUNS/cf-fm-tokens-xattn-long/checkpoints" "$RUNS/cf-fm-tokens-xattn/checkpoints")
CF_MEANPOOL_DIRS=("$RUNS/cf-fm-meanpool-adagn-long/checkpoints" "$RUNS/cf-fm-meanpool-adagn/checkpoints")
: "${OUT:=/orcd/pool/007/pablomer/outputs/counterfactual_fm/results/recon_mse.csv}"
: "${N:=256}"
: "${NUM_STEPS:=250}"
: "${MMAP_STORE:=1}"          # 1 = memmap the store; the eval reads ~66 MB, not 17 GB
: "${FIRST_CHUNK:=16}"          # early three-way report after this many galaxies
: "${REPORT_AT:=16,32,64,128,256,512,1024,2048}"
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
# NOTE the `|| true`: a candidate dir that exists but holds no best-*.ckpt leaves the
# glob unexpanded, so `ls` exits 2 and `set -eo pipefail` would abort the whole eval.
resolve_best() {
    local dir
    for dir in "$@"; do
        [ -d "$dir" ] || continue
        ls -t "$dir"/best-*.ckpt 2>/dev/null | head -n1 || true
    done | head -n1 || true
}

CF_ARGS=()
TOK=$(resolve_best "${CF_TOKENS_DIRS[@]}" || true)
if [ -n "$TOK" ]; then
    CF_ARGS+=(--cf-checkpoint "cf-fm-tokens:$TOK")
    echo "[eval] tokens ckpt=$TOK"
else
    echo "[eval] WARNING: no tokens best-*.ckpt in ${CF_TOKENS_DIRS[*]} — CF tokens row will be MISSING"
fi
MP=$(resolve_best "${CF_MEANPOOL_DIRS[@]}" || true)
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
    ${MMAP_STORE:+--mmap-store} \
    --first-chunk "$FIRST_CHUNK" \
    --report-at "$REPORT_AT" \
    --discord-webhook "$WEBHOOK" \
    "${CF_ARGS[@]}"

echo "Eval done at $(date) -> $OUT"
