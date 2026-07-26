#!/bin/bash
#SBATCH --job-name=recon-mse-scale
#SBATCH --partition=pg_mki_aryeh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/home/pablomer/orcd/pool/tess-generative/galaxy_images/galaxy_model/scale_ablation/logs/%x_%j.out
#SBATCH --error=/home/pablomer/orcd/pool/tess-generative/galaxy_images/galaxy_model/scale_ablation/logs/%x_%j.err
#SBATCH --open-mode=append
#
# C0 (intact) reconstruction MSE for the scale-ablation curve, one call for all models
# with IDENTICAL x0 noise per anchor (noise_seed + batch_idx protocol).
#
# EVAL SET: a NEW 256-anchor manifest drawn from the HOLDOUT-FILTERED seed-42 val split
# (kept=98,275 -> val=4,913; --config scale_1000.json only determines this loader — the
# train-subset key does not touch the val split). These galaxies are genuinely held out
# for the four scale arms (enforced at training time by _apply_train_subset).
# The base-87k / fm-control-75k / constlr-35k rows trained WITHOUT the holdout, so ~95%
# of these eval galaxies were in their training sets: their MSE is optimistic. That bias
# is CONSERVATIVE for the scaling claim (it can only widen the arms-vs-base gap), and the
# rows are labeled accordingly. Do NOT reuse diffusion_ablation's manifest here — the val
# split differs (kept_size 103,741 vs 98,275) and the script hard-errors on mismatch.

set -eo pipefail

REPO="/home/pablomer/orcd/pool/tess-generative"
ABL="${REPO}/galaxy_images/galaxy_model/scale_ablation"
FROZEN="/orcd/pool/007/pablomer/checkpoints_new/scale_ablation/frozen"
DIFF_FROZEN="/orcd/pool/007/pablomer/checkpoints_new/diffusion_ablation/frozen"
PAPER_BASE="/orcd/pool/007/pablomer/checkpoints_new/base/snapshot.ckpt"
CONSTLR="/orcd/pool/007/pablomer/checkpoints_new/lr_schedule_probe/best/base-constlr-37k5/best-epoch=91-step=35000.ckpt"
mkdir -p "${ABL}/logs" "${ABL}/results"

WEBHOOK="${DISCORD_WEBHOOK:-https://discord.com/api/webhooks/1500242707522519183/NqqAxuo5n0Ex4X81vMcq7gDOjrP3oAURAVZtgrZT5CramGNfBGPxuEvC2NnxwYBkfs9j}"
_ping() { curl -sS -X POST -H 'Content-Type: application/json' -d "{\"content\": \"$1\"}" "${WEBHOOK}" >/dev/null 2>&1 || true; }
_ping "recon-mse-scale 🚀 start · 7 models x 256 anchors x 250 Euler steps · job ${SLURM_JOB_ID:-?}"
trap '_ping "recon-mse-scale $([ $? -eq 0 ] && echo ✅ || echo ❌) · job ${SLURM_JOB_ID:-?}"' EXIT

source ~/.bashrc
conda activate torchenv
cd "${REPO}"
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

srun --unbuffered python -u -m galaxy_images.galaxy_model.diffusion_ablation.eval_recon_mse \
  --config "${ABL}/configs/scale_1000.json" \
  --manifest "${ABL}/results/recon_eval_manifest.json" \
  --out "${ABL}/results/recon_mse_scale.csv" \
  --n 256 --noise-seed 1234 --num-steps 250 \
  --skip-paper \
  --fm-checkpoint "scale-1000:${FROZEN}/scale_1000/scale-1000-75k.ckpt" \
  --fm-checkpoint "scale-3162:${FROZEN}/scale_3162/scale-3162-75k.ckpt" \
  --fm-checkpoint "scale-10000:${FROZEN}/scale_10000/scale-10000-75k.ckpt" \
  --fm-checkpoint "scale-31622:${FROZEN}/scale_31622/scale-31622-75k.ckpt" \
  --fm-checkpoint "fm-control-75k:${DIFF_FROZEN}/fm-control-75k-fixed.ckpt" \
  --fm-checkpoint "base-87k:${PAPER_BASE}" \
  --fm-checkpoint "constlr-35k:${CONSTLR}" \
  --discord-webhook "${WEBHOOK}" \
  --ping-prefix "recon-mse-scale"
