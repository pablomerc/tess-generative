#!/bin/bash
#SBATCH --job-name=recon-mse-scale2
#SBATCH --partition=pg_mki_aryeh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/home/pablomer/orcd/pool/tess-generative/galaxy_images/galaxy_model/scale_ablation/logs/%x_%j.out
#SBATCH --error=/home/pablomer/orcd/pool/tess-generative/galaxy_images/galaxy_model/scale_ablation/logs/%x_%j.err
#SBATCH --open-mode=append
#
# Part 2 of the C0 recon-MSE eval. Job 18867153 TIMED OUT at its 2 h wall having finished
# only scale-1000 and scale-3162: ~93 min of that went to the initial 17 GB RAM48 load
# (cold Ceph read; manifest written 19:33 for an 18:00 start), leaving ~13 min per model.
# Five models remain -> ~93 + 5x13 = ~160 min. 6 h gives comfortable margin.
#
# Reuses results/recon_eval_manifest.json UNCHANGED, so this run is pinned to the exact
# same 256 held-out anchors and the same per-batch x0 noise seeds as part 1 — the script
# hard-errors on any mismatch. Concatenating the two CSVs is therefore exactly equivalent
# to one run (report_scale_results.py globs recon_mse_scale*.csv and dedups).

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
_ping "recon-mse-scale2 🚀 start · 5 remaining models (10k, 31.6k, fm-control, base-87k, constlr) · job ${SLURM_JOB_ID:-?}"
trap '_ping "recon-mse-scale2 $([ $? -eq 0 ] && echo ✅ || echo ❌) · job ${SLURM_JOB_ID:-?}"' EXIT

source ~/.bashrc
conda activate torchenv
cd "${REPO}"
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

srun --unbuffered python -u -m galaxy_images.galaxy_model.diffusion_ablation.eval_recon_mse \
  --config "${ABL}/configs/scale_1000.json" \
  --manifest "${ABL}/results/recon_eval_manifest.json" \
  --out "${ABL}/results/recon_mse_scale_part2.csv" \
  --n 256 --noise-seed 1234 --num-steps 250 \
  --skip-paper \
  --fm-checkpoint "scale-10000:${FROZEN}/scale_10000/scale-10000-75k.ckpt" \
  --fm-checkpoint "scale-31622:${FROZEN}/scale_31622/scale-31622-75k.ckpt" \
  --fm-checkpoint "fm-control-75k:${DIFF_FROZEN}/fm-control-75k-fixed.ckpt" \
  --fm-checkpoint "base-87k:${PAPER_BASE}" \
  --fm-checkpoint "constlr-35k:${CONSTLR}" \
  --discord-webhook "${WEBHOOK}" \
  --ping-prefix "recon-mse-scale2"
