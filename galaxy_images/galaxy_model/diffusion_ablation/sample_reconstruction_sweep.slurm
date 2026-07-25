#!/bin/bash
#SBATCH --job-name=ddpm-recon-sweep
#SBATCH --partition=mit_normal_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/home/pablomer/orcd/pool/tess-generative/galaxy_images/galaxy_model/diffusion_ablation/logs/%x_%j.out
#SBATCH --error=/home/pablomer/orcd/pool/tess-generative/galaxy_images/galaxy_model/diffusion_ablation/logs/%x_%j.err
#SBATCH --open-mode=append
#
# Optional overrides:
#   sbatch --export=ALL,NUM_IMAGES=12,NUM_STEPS='10 25 50 100 250',ETAS='0 0.5 1' \
#     galaxy_images/galaxy_model/diffusion_ablation/sample_reconstruction_sweep.slurm

set -eo pipefail

REPO="/home/pablomer/orcd/pool/tess-generative"
ABLATION_DIR="${REPO}/galaxy_images/galaxy_model/diffusion_ablation"
CHECKPOINT="${CHECKPOINT:-/home/pablomer/orcd/pool/checkpoints_new/diffusion_ablation/best/ddpm-eps-ram48-h200-75k/best-epoch=72-step=28000_copy.ckpt}"
CONFIG="${CONFIG:-${ABLATION_DIR}/configs/neighbors_ddpm_eps.json}"
OUTPUT_DIR="${OUTPUT_DIR:-${ABLATION_DIR}/results/reconstruction_sweep/best-step28000}"
NUM_IMAGES="${NUM_IMAGES:-8}"
NUM_STEPS="${NUM_STEPS:-25 50 100 250}"
ETAS="${ETAS:-0 0.5 1}"
NOISE_SEED="${NOISE_SEED:-1234}"

mkdir -p "${ABLATION_DIR}/logs" "${OUTPUT_DIR}"

WEBHOOK="${DISCORD_WEBHOOK:-https://discord.com/api/webhooks/1500242707522519183/NqqAxuo5n0Ex4X81vMcq7gDOjrP3oAURAVZtgrZT5CramGNfBGPxuEvC2NnxwYBkfs9j}"
_ping() {
    curl -sS -X POST -H 'Content-Type: application/json' \
        -d "{\"content\": \"$1\"}" "${WEBHOOK}" >/dev/null 2>&1 || true
}

_ping "ddpm-recon-sweep 🚀 start · ${NUM_IMAGES} images · steps=${NUM_STEPS} · eta=${ETAS} · job ${SLURM_JOB_ID:-?}"
cleanup() {
    local rc=$?
    if [[ ${rc} -eq 0 ]]; then
        _ping "ddpm-recon-sweep ✅ finished · output=\`${OUTPUT_DIR}\` · job ${SLURM_JOB_ID:-?}"
    else
        _ping "ddpm-recon-sweep ❌ failed (exit ${rc}) · job ${SLURM_JOB_ID:-?}"
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

# shellcheck disable=SC2086
python -u -m galaxy_images.galaxy_model.diffusion_ablation.sample_reconstruction_sweep \
    --checkpoint "${CHECKPOINT}" \
    --config "${CONFIG}" \
    --output-dir "${OUTPUT_DIR}" \
    --num-images "${NUM_IMAGES}" \
    --num-steps ${NUM_STEPS} \
    --eta ${ETAS} \
    --noise-seed "${NOISE_SEED}"
