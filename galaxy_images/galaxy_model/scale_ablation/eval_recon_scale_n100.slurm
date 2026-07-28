#!/bin/bash
#SBATCH --job-name=recon-mse-scale100
#SBATCH --partition=pg_mki_aryeh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/home/pablomer/orcd/pool/tess-generative/galaxy_images/galaxy_model/scale_ablation/logs/%x_%j.out
#SBATCH --error=/home/pablomer/orcd/pool/tess-generative/galaxy_images/galaxy_model/scale_ablation/logs/%x_%j.err
#SBATCH --open-mode=append
#
# C0 (intact) reconstruction MSE for the EXTRA n=100 rung.
#
# Reuses results/recon_eval_manifest.json UNCHANGED, so this is pinned to the exact same 256
# held-out anchors and the same per-batch x0 noise seeds as parts 1 and 2 — eval_recon_mse.py
# hard-errors on any mismatch. Its output therefore concatenates with the existing CSVs as if
# all arms had been run in a single job (report_scale_results.py globs recon_mse_scale*.csv
# and dedups on model_label × anchor_survey).
#
# --config scale_1000.json is deliberate and matches parts 1/2: the config only determines the
# val-split loader, and the train-subset key does not touch the val split. Using scale_100.json
# here would produce the same eval set but gratuitously diverge from the earlier invocations.
#
# One model, so the ~90 min cold RAM48 load dominates: 4 h wall is ample.

set -eo pipefail

REPO="/home/pablomer/orcd/pool/tess-generative"
ABL="${REPO}/galaxy_images/galaxy_model/scale_ablation"
FROZEN="/orcd/pool/007/pablomer/checkpoints_new/scale_ablation/frozen"
mkdir -p "${ABL}/logs" "${ABL}/results"

CKPT="${FROZEN}/scale_100/scale-100-75k.ckpt"

WEBHOOK="${DISCORD_WEBHOOK:-https://discord.com/api/webhooks/1500242707522519183/NqqAxuo5n0Ex4X81vMcq7gDOjrP3oAURAVZtgrZT5CramGNfBGPxuEvC2NnxwYBkfs9j}"
_ping() { curl -sS -X POST -H 'Content-Type: application/json' -d "{\"content\": \"$1\"}" "${WEBHOOK}" >/dev/null 2>&1 || true; }
_ping "recon-mse-scale100 🚀 start · n=100 arm · 256 held-out anchors × 250 Euler steps · job ${SLURM_JOB_ID:-?}"
trap '_ping "recon-mse-scale100 $([ $? -eq 0 ] && echo ✅ || echo ❌) · job ${SLURM_JOB_ID:-?}"' EXIT

source ~/.bashrc
conda activate torchenv
cd "${REPO}"
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

test -f "${CKPT}" || { echo "FATAL: frozen checkpoint missing: ${CKPT}"; exit 1; }

srun --unbuffered python -u -m galaxy_images.galaxy_model.diffusion_ablation.eval_recon_mse \
  --config "${ABL}/configs/scale_1000.json" \
  --manifest "${ABL}/results/recon_eval_manifest.json" \
  --out "${ABL}/results/recon_mse_scale_n100.csv" \
  --n 256 --noise-seed 1234 --num-steps 250 \
  --skip-paper \
  --fm-checkpoint "scale-100:${CKPT}" \
  --discord-webhook "${WEBHOOK}" \
  --ping-prefix "recon-mse-scale100"
