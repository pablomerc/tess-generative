#!/bin/bash
#SBATCH --job-name=psf-traversal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:30:00
#SBATCH --partition=mi2104x
#SBATCH --output=/work1/jeroenaudenaert/pablomer/logs/psf_traversal_%j.out
#SBATCH --error=/work1/jeroenaudenaert/pablomer/logs/psf_traversal_%j.err
#SBATCH --open-mode=append

set -eo pipefail

mkdir -p /work1/jeroenaudenaert/pablomer/logs

# --- Environment ---
source ~/.bashrc
conda activate torchenv

# --- ROCm workaround ---
export TORCH_BLAS_PREFER_HIPBLASLT=0

# --- Output subfolder ---
RUN_DIR="/work1/jeroenaudenaert/pablomer/tess-generative/galaxy_images/galaxy_model/latent_traversal/psf_traversal_v2"
mkdir -p "${RUN_DIR}"

# --- Run ---
cd /work1/jeroenaudenaert/pablomer/tess-generative
export PYTHONPATH=/work1/jeroenaudenaert/pablomer/tess-generative:$PYTHONPATH

# 20 evenly-spaced target indices across the dataset (step=5000)
TARGET_INDICES=(0 5000 10000 15000 20000 25000 30000 35000 40000 45000 50000 55000 60000 65000 70000 75000 80000 85000 90000 95000)

for TARGET_IDX in "${TARGET_INDICES[@]}"; do
    echo "--- Running traversal for target_idx=${TARGET_IDX} ---"
    python -u galaxy_images/galaxy_model/latent_traversal/psf_conditioning_traversal.py \
        --target-idx "${TARGET_IDX}" \
        --n-examples 100000 \
        --output-dir "${RUN_DIR}"
done

echo "All 20 targets done at $(date)"

DISCORD_WEBHOOK="https://discord.com/api/webhooks/1488692651334177071/8b8KvACfQIVYCNY3ovee04BixCEWiqbqp1iQk4z9sXHlgR29kMkGIjl1pahV5uEPSbxe"
curl -s -F "content=**Job \`${SLURM_JOB_ID}\` done**: psf-conditioning-traversal (20 targets)" \
    "${DISCORD_WEBHOOK}" || true

for FIG in "${RUN_DIR}"/*.png; do
    [ -f "${FIG}" ] && curl -s -F "file=@${FIG}" "${DISCORD_WEBHOOK}" || true
done
