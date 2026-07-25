#!/bin/bash
#SBATCH --job-name=scale-snapshot
#SBATCH --partition=pg_mki_aryeh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=23:00:00
#SBATCH --output=/home/pablomer/orcd/pool/tess-generative/galaxy_images/galaxy_model/scale_ablation/logs/%x_%j.out
#SBATCH --error=/home/pablomer/orcd/pool/tess-generative/galaxy_images/galaxy_model/scale_ablation/logs/%x_%j.err
#
# CPU-only checkpoint archiver. No --gres, so this lands on node4701/4702 (the CPU-only
# nodes in pg_mki_aryeh) and never competes with the training arms for H200s.
#
#   sbatch galaxy_images/galaxy_model/scale_ablation/snapshot_checkpoints.slurm
#
# Exists because train.py's periodic ModelCheckpoint runs save_top_k=1, so each new
# `latest-step=step=N.ckpt` DELETES the previous one. Without this poller an R^2-vs-step
# curve is impossible after the fact.

set -eo pipefail

REPO="/home/pablomer/orcd/pool/tess-generative"
mkdir -p "${REPO}/galaxy_images/galaxy_model/scale_ablation/logs"

WEBHOOK="${DISCORD_WEBHOOK:-https://discord.com/api/webhooks/1500242707522519183/NqqAxuo5n0Ex4X81vMcq7gDOjrP3oAURAVZtgrZT5CramGNfBGPxuEvC2NnxwYBkfs9j}"
_ping() { curl -sS -X POST -H 'Content-Type: application/json' -d "{\"content\": \"$1\"}" "${WEBHOOK}" >/dev/null 2>&1 || true; }

_ping "scale-snapshot 📸 archiver started · job ${SLURM_JOB_ID:-?} · ladder 5k/10k/15k/25k/40k/75k per arm"
trap '_ping "scale-snapshot 📸 archiver exited (rc=$?) · job ${SLURM_JOB_ID:-?}"' EXIT

source ~/.bashrc
conda activate torchenv
cd "${REPO}"
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"

srun --unbuffered python -u galaxy_images/galaxy_model/scale_ablation/snapshot_checkpoints.py \
  --interval 240
