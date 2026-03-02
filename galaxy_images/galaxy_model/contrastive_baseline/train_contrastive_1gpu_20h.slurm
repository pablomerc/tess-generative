#!/bin/bash
#SBATCH --job-name=contrastive-1gpu
#SBATCH --qos=shared-if-available
#SBATCH --partition=vision-shared-v100
#SBATCH --account=vision-billf
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=20:00:00
#SBATCH --gres=gpu:4
#SBATCH --output=/data/vision/billf/scratch/pablomer/projects/tess-generative/galaxy_images/galaxy_model/contrastive_baseline/logs/job_%j.out
#SBATCH --error=/data/vision/billf/scratch/pablomer/projects/tess-generative/galaxy_images/galaxy_model/contrastive_baseline/logs/job_%j.err
#SBATCH --open-mode=append

set -euo pipefail

cd /data/vision/billf/scratch/pablomer/projects/tess-generative
mkdir -p galaxy_images/galaxy_model/contrastive_baseline/logs
mkdir -p /data/vision/billf/scratch/pablomer/outputs/contrastive_baseline
mkdir -p /tmp/matplotlib /tmp/numba_cache

source /data/vision/billf/scratch/pablomer/miniconda3/etc/profile.d/conda.sh
conda activate py310-torch

# Make cache dirs writable on cluster nodes.
export MPLCONFIGDIR=/tmp/matplotlib
export NUMBA_CACHE_DIR=/tmp/numba_cache

# Use 1 GPU in Trainer.
export N_DEVICES=4

# Load wandb key without printing it.
set -a
source /data/vision/billf/scratch/pablomer/.secrets/wandb.env
set +a

srun python -u galaxy_images/galaxy_model/contrastive_baseline/train_dual_encoder_contrastive.py

echo "Job completed at $(date)"
