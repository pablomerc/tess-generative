#!/bin/bash
#SBATCH --job-name=ours-i4x4
#SBATCH --qos=vision-billf-main
#SBATCH --partition=vision-billf-v100
#SBATCH --account=vision-billf
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=/data/vision/billf/scratch/pablomer/projects/tess-generative/galaxy_images/galaxy_model/downstream_evaluation/logs/job_%j.out
#SBATCH --error=/data/vision/billf/scratch/pablomer/projects/tess-generative/galaxy_images/galaxy_model/downstream_evaluation/logs/job_%j.err
#SBATCH --open-mode=append

set -euo pipefail

cd /data/vision/billf/scratch/pablomer/projects/tess-generative
mkdir -p galaxy_images/galaxy_model/downstream_evaluation/logs
mkdir -p /tmp/matplotlib /tmp/numba_cache

source /data/vision/billf/scratch/pablomer/miniconda3/etc/profile.d/conda.sh
conda activate py310-torch

export MPLCONFIGDIR=/tmp/matplotlib
export NUMBA_CACHE_DIR=/tmp/numba_cache

srun python -u galaxy_images/galaxy_model/downstream_evaluation/predict_all_ours_instrument4x4.py \
    --suffix ours_instrument4x4 \
    --output-dir galaxy_images/galaxy_model/downstream_evaluation/4x4

echo "Job completed at $(date)"
