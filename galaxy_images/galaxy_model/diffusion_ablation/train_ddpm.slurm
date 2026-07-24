#!/bin/bash
#SBATCH --job-name=diff-ablation
#SBATCH --partition=pg_mki_aryeh
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:h200:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=/home/pablomer/orcd/pool/tess-generative/galaxy_images/galaxy_model/diffusion_ablation/logs/%x_%j.out
#SBATCH --error=/home/pablomer/orcd/pool/tess-generative/galaxy_images/galaxy_model/diffusion_ablation/logs/%x_%j.err
#SBATCH --open-mode=append
#
# Shared trainer for both ablation arms. Parameterize with:
#   sbatch --export=ALL,CONFIG_PATH=.../neighbors_ddpm_eps.json \
#          diffusion_ablation/train_ddpm.slurm
#   sbatch --export=ALL,CONFIG_PATH=.../neighbors_fm_control.json \
#          diffusion_ablation/train_ddpm.slurm
#
# Optional: EXTRA_SET_ARGS='--set trainer.num_steps=200 --set wandb.enabled=false'

set -eo pipefail

REPO="/home/pablomer/orcd/pool/tess-generative"
ABL_DIR="${REPO}/galaxy_images/galaxy_model/diffusion_ablation"
mkdir -p "${ABL_DIR}/logs"

: "${CONFIG_PATH:?set CONFIG_PATH to a diffusion_ablation JSON config}"
EXTRA_SET_ARGS="${EXTRA_SET_ARGS:-}"

WEBHOOK="${DISCORD_WEBHOOK:-https://discord.com/api/webhooks/1500242707522519183/NqqAxuo5n0Ex4X81vMcq7gDOjrP3oAURAVZtgrZT5CramGNfBGPxuEvC2NnxwYBkfs9j}"
_ping() { curl -sS -X POST -H 'Content-Type: application/json' -d "{\"content\": \"$1\"}" "${WEBHOOK}" >/dev/null 2>&1 || true; }

CONFIG_BASENAME="$(basename "${CONFIG_PATH}")"
_ping "diff-ablation 🚀 start · config=\`${CONFIG_BASENAME}\` · job ${SLURM_JOB_ID:-?} · host ${SLURMD_NODENAME:-?}"

cleanup() {
  local rc=$?
  if [[ ${rc} -eq 0 ]]; then
    _ping "diff-ablation ✅ finished · config=\`${CONFIG_BASENAME}\` · job ${SLURM_JOB_ID:-?}"
  else
    _ping "diff-ablation ❌ failed (exit ${rc}) · config=\`${CONFIG_BASENAME}\` · job ${SLURM_JOB_ID:-?}"
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
# shellcheck disable=SC2086
srun python -u -m galaxy_images.galaxy_model.train --config "${CONFIG_PATH}" ${EXTRA_SET_ARGS}
