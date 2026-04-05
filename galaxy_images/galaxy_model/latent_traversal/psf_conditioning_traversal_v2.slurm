#!/bin/bash
#SBATCH --job-name=psf-traversal-v3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=03:00:00
#SBATCH --partition=mi2104x
#SBATCH --output=/work1/jeroenaudenaert/pablomer/logs/psf_traversal_v2_%j.out
#SBATCH --error=/work1/jeroenaudenaert/pablomer/logs/psf_traversal_v2_%j.err
#SBATCH --open-mode=append

set -eo pipefail

mkdir -p /work1/jeroenaudenaert/pablomer/logs

# --- Environment ---
source ~/.bashrc
conda activate torchenv

# --- ROCm workaround ---
export TORCH_BLAS_PREFER_HIPBLASLT=0

# --- Output subfolder ---
RUN_DIR="/work1/jeroenaudenaert/pablomer/tess-generative/galaxy_images/galaxy_model/latent_traversal/psf_traversal_v5"
mkdir -p "${RUN_DIR}"

# --- Run ---
cd /work1/jeroenaudenaert/pablomer/tess-generative
export PYTHONPATH=/work1/jeroenaudenaert/pablomer/tess-generative:$PYTHONPATH

# 20 evenly-spaced target indices across the dataset (step=5000)
TARGET_INDICES=(0 5003 10006 15009 20012 25015 30018 35021 40024 45027 50030 55033 60036 65039 70042 75045 80048 85051 90054 95057)

for TARGET_IDX in "${TARGET_INDICES[@]}"; do
    echo "--- Running v2 traversal for target_idx=${TARGET_IDX} (bucket-size=5) ---"
    python -u galaxy_images/galaxy_model/latent_traversal/psf_conditioning_traversal_v2.py \
        --target-idx "${TARGET_IDX}" \
        --bucket-size 5 \
        --output-dir "${RUN_DIR}"

    echo "--- Running v2 traversal for target_idx=${TARGET_IDX} (bucket-size=15) ---"
    python -u galaxy_images/galaxy_model/latent_traversal/psf_conditioning_traversal_v2.py \
        --target-idx "${TARGET_IDX}" \
        --bucket-size 15 \
        --output-dir "${RUN_DIR}"
done

echo "All 20 targets done at $(date)"

DISCORD_WEBHOOK="https://discord.com/api/webhooks/1488692651334177071/8b8KvACfQIVYCNY3ovee04BixCEWiqbqp1iQk4z9sXHlgR29kMkGIjl1pahV5uEPSbxe"
curl -s -F "content=**Job \`${SLURM_JOB_ID}\` done**: psf-conditioning-traversal-v3 (20 targets, kb5+kb15)" \
    "${DISCORD_WEBHOOK}" || true

for FIG in "${RUN_DIR}"/*.png; do
    [ -f "${FIG}" ] && curl -s -F "file=@${FIG}" "${DISCORD_WEBHOOK}" || true
done
