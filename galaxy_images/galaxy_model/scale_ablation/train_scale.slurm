#!/bin/bash
#SBATCH --job-name=scale-ablation
#SBATCH --partition=pg_mki_aryeh
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:h200:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=/home/pablomer/orcd/pool/tess-generative/galaxy_images/galaxy_model/scale_ablation/logs/%x_%j.out
#SBATCH --error=/home/pablomer/orcd/pool/tess-generative/galaxy_images/galaxy_model/scale_ablation/logs/%x_%j.err
#SBATCH --open-mode=append
#
# Shared trainer for every rung of the data-scale ablation. Parameterize with:
#   sbatch --job-name=scale-1000 --export=ALL,CONFIG_PATH=<repo>/galaxy_images/galaxy_model/scale_ablation/configs/scale_1000.json \
#          galaxy_images/galaxy_model/scale_ablation/train_scale.slurm
#
# Optional: EXTRA_SET_ARGS='--set trainer.num_steps=200 --set wandb.enabled=false --set trainer.monitor_metric=val/loss'
#
# Measured cost: 75k steps at ~2.7 optimizer steps/s on 4xH200 = ~7.7 h, well inside the
# 24 h wall, so no self-chaining is needed. Resume (if ever required) is manual:
#   EXTRA_SET_ARGS='--set run.resume_from=<...latest-step=step=NNNNN.ckpt>'
# There is no last.ckpt, and note the doubled 'step=' in real filenames.

set -eo pipefail

REPO="/home/pablomer/orcd/pool/tess-generative"
ABL_DIR="${REPO}/galaxy_images/galaxy_model/scale_ablation"
mkdir -p "${ABL_DIR}/logs"

: "${CONFIG_PATH:?set CONFIG_PATH to a scale_ablation JSON config}"
EXTRA_SET_ARGS="${EXTRA_SET_ARGS:-}"

WEBHOOK="${DISCORD_WEBHOOK:-https://discord.com/api/webhooks/1500242707522519183/NqqAxuo5n0Ex4X81vMcq7gDOjrP3oAURAVZtgrZT5CramGNfBGPxuEvC2NnxwYBkfs9j}"
_ping() { curl -sS -X POST -H 'Content-Type: application/json' -d "{\"content\": \"$1\"}" "${WEBHOOK}" >/dev/null 2>&1 || true; }

CONFIG_BASENAME="$(basename "${CONFIG_PATH}")"
_ping "scale-ablation 🚀 start · config=\`${CONFIG_BASENAME}\` · job ${SLURM_JOB_ID:-?} · host ${SLURMD_NODENAME:-?}"

cleanup() {
  local rc=$?
  if [[ ${rc} -eq 0 ]]; then
    _ping "scale-ablation ✅ finished · config=\`${CONFIG_BASENAME}\` · job ${SLURM_JOB_ID:-?}"
  else
    _ping "scale-ablation ❌ failed (exit ${rc}) · config=\`${CONFIG_BASENAME}\` · job ${SLURM_JOB_ID:-?}"
  fi
}
trap cleanup EXIT

source ~/.bashrc
conda activate torchenv

cd "${REPO}"
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export TORCH_BLAS_PREFER_HIPBLASLT=0

# WandB: online if authenticated (env key OR a ~/.netrc wandb login), else offline.
if [[ -n "${WANDB_API_KEY:-}" ]] || grep -q "api.wandb.ai" ~/.netrc 2>/dev/null; then
    export WANDB_MODE=online
else
    export WANDB_MODE="${WANDB_MODE:-offline}"
fi
echo "[wandb] WANDB_MODE=${WANDB_MODE}"
echo "[train] CONFIG_PATH=${CONFIG_PATH}"
echo "[train] EXTRA_SET_ARGS=${EXTRA_SET_ARGS}"

# DDP launch contract (PL 2.6.1 + SLURM): srun sets SLURM_NTASKS, so Lightning's
# SLURMEnvironment takes over and does NOT self-spawn ranks — srun itself must launch
# them. Hence --ntasks-per-node=4 above (NOT --ntasks=1, which silently trains on 1 GPU
# while sharding data 4 ways; and NOT --ntasks=4 alone, which PL rejects without
# SLURM_NTASKS_PER_NODE). Verify in the log: "Starting with 4 processes".
#
# Also verify in the log, specific to this ablation:
#   [downstream-holdout] matched anchors: 5,466 / 103,741
#   [data] after exclusions: kept=98,275 | val_ratio=0.05 -> train=93,362, val=4,913
#   [train-subset] scale_<N>.json: distinct anchors=<N> ... tiled to 93,362 items
# The tiled count MUST be 93,362 in every arm — that is what keeps steps/epoch at 364
# and the (epoch-driven) LR waveform identical across data scales.
#
# --unbuffered: without it srun block-buffers task stdout, so the startup prints
# (dataset load, holdout, subset tiling) sit invisible for many minutes and a slow
# RAM48 load is indistinguishable from a hang. Learned the hard way on job 18790012.
# shellcheck disable=SC2086
srun --unbuffered python -u -m galaxy_images.galaxy_model.train --config "${CONFIG_PATH}" ${EXTRA_SET_ARGS}
