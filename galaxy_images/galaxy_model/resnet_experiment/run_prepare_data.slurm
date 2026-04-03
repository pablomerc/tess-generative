#!/bin/bash
# ==============================================================================
# Preprocess HSC parquet files → resnet_data.h5
# ~396 parquet files, ~101k rows. CPU-only, ~1-2h.
# ==============================================================================
#SBATCH --job-name=resnet-prep
#SBATCH --qos=vision-billf-main
#SBATCH --partition=vision-billf-v100
#SBATCH --account=vision-billf
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:0
#SBATCH --output=logs/job_%j.out
#SBATCH --error=logs/job_%j.err

set -e

REPO_ROOT="/data/vision/billf/scratch/pablomer/projects/tess-generative"
EXPERIMENT_DIR="${REPO_ROOT}/galaxy_images/galaxy_model/resnet_experiment"

source /data/vision/billf/scratch/pablomer/miniconda3/etc/profile.d/conda.sh
conda activate py310-torch

cd "${EXPERIMENT_DIR}"

echo "================================================================"
echo "Preparing ResNet training data from parquet files"
echo "================================================================"
python prepare_data.py

echo ""
echo "================================================================"
echo "Done. Output: ${EXPERIMENT_DIR}/resnet_data.h5"
echo "================================================================"

DISCORD_WEBHOOK="https://discord.com/api/webhooks/1488692651334177071/8b8KvACfQIVYCNY3ovee04BixCEWiqbqp1iQk4z9sXHlgR29kMkGIjl1pahV5uEPSbxe"
curl -s -F "content=✅ **Job \`${SLURM_JOB_ID}\` done**: resnet-prep — resnet_data.h5 ready" \
    "${DISCORD_WEBHOOK}" || true
