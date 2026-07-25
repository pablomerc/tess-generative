#!/bin/bash
#SBATCH --job-name=recon-mse
#SBATCH --partition=mit_normal_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/home/pablomer/orcd/pool/tess-generative/galaxy_images/galaxy_model/diffusion_ablation/logs/%x_%j.out
#SBATCH --error=/home/pablomer/orcd/pool/tess-generative/galaxy_images/galaxy_model/diffusion_ablation/logs/%x_%j.err
#SBATCH --open-mode=append
#
# Matched recon-MSE grid (steps x eta); pings Discord after EVERY setting and
# posts the final table + writes results/<stem>.{csv,pdf,tex}.
#   sbatch --export=ALL,FM_CKPT=...,DDPM_CKPT=...,OUT_CSV=...,TAG=...,STEPS='25 50 100 250',ETAS='0 0.5 1' eval_recon_mse.slurm

set -eo pipefail

REPO="/home/pablomer/orcd/pool/tess-generative"
ABL="${REPO}/galaxy_images/galaxy_model/diffusion_ablation"
mkdir -p "${ABL}/logs" "${ABL}/results"

FM_CKPT="${FM_CKPT:?set FM_CKPT}"
DDPM_CKPT="${DDPM_CKPT:?set DDPM_CKPT}"
OUT_CSV="${OUT_CSV:-${ABL}/results/recon_mse_grid.csv}"
TAG="${TAG:-recon-mse}"
N="${N:-256}"
STEPS="${STEPS:-250}"
ETAS="${ETAS:-0.0 1.0}"
TABLE_STEM="${TABLE_STEM:-${OUT_CSV%.csv}}"

WEBHOOK="${DISCORD_WEBHOOK:-https://discord.com/api/webhooks/1500242707522519183/NqqAxuo5n0Ex4X81vMcq7gDOjrP3oAURAVZtgrZT5CramGNfBGPxuEvC2NnxwYBkfs9j}"
_ping() { curl -sS -X POST -H 'Content-Type: application/json' -d "$1" "${WEBHOOK}" >/dev/null 2>&1 || true; }

_ping "{\"content\": \"recon-mse 🚀 start · ${TAG} · n=${N} · steps ∈ {${STEPS}} · η ∈ {${ETAS}} · job ${SLURM_JOB_ID:-?} (partial pings after each setting)\"}"
cleanup() {
  local rc=$?
  [[ ${rc} -ne 0 ]] && _ping "{\"content\": \"recon-mse ❌ failed (exit ${rc}) · ${TAG} · job ${SLURM_JOB_ID:-?} — see logs/recon-mse_${SLURM_JOB_ID:-?}.err\"}"
}
trap cleanup EXIT

source ~/.bashrc
conda activate torchenv
cd "${REPO}"
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export TORCH_BLAS_PREFER_HIPBLASLT=0

STEP_FLAGS=""; for s in ${STEPS}; do STEP_FLAGS+=" --num-steps ${s}"; done
ETA_FLAGS="";  for e in ${ETAS};  do ETA_FLAGS+=" --eta ${e}";        done
CLIP_FLAG=""; [[ -n "${CLIP_RANGE:-}" ]] && CLIP_FLAG="--ddpm-clip-range ${CLIP_RANGE}"

# shellcheck disable=SC2086
python -u -m galaxy_images.galaxy_model.diffusion_ablation.eval_recon_mse \
  --fm-checkpoint "fm-control:${FM_CKPT}" \
  --ddpm-checkpoint "${DDPM_CKPT}" \
  --config "${ABL}/configs/neighbors_ddpm_eps.json" \
  --n "${N}" ${STEP_FLAGS} ${ETA_FLAGS} ${CLIP_FLAG} \
  --discord-webhook "${WEBHOOK}" \
  --ping-prefix "${TAG}" \
  --skip-paper \
  --out "${OUT_CSV}"

# Final table: PDF + tex + Discord post.
PAYLOAD="$(mktemp)"
python -u -m galaxy_images.galaxy_model.diffusion_ablation.make_table_pdf \
  --kind mse --csv "${OUT_CSV}" --out-stem "${TABLE_STEM}" \
  --payload-out "${PAYLOAD}" --tag "${TAG}"
curl -sS -X POST -H 'Content-Type: application/json' -d @"${PAYLOAD}" "${WEBHOOK}" >/dev/null 2>&1 || true
rm -f "${PAYLOAD}"
echo "[eval] done; table at ${TABLE_STEM}.pdf"
