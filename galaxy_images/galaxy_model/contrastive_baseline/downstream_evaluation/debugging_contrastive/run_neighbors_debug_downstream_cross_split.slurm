#!/bin/bash
#SBATCH --job-name=cnt-nb-xsp
#SBATCH --qos=vision-billf-main
#SBATCH --partition=vision-billf-v100
#SBATCH --account=vision-billf
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=/data/vision/billf/scratch/pablomer/projects/tess-generative/galaxy_images/galaxy_model/contrastive_baseline/logs/job_%j.out
#SBATCH --error=/data/vision/billf/scratch/pablomer/projects/tess-generative/galaxy_images/galaxy_model/contrastive_baseline/logs/job_%j.err
#SBATCH --open-mode=append

set -euo pipefail

cd /data/vision/billf/scratch/pablomer/projects/tess-generative
mkdir -p galaxy_images/galaxy_model/contrastive_baseline/logs
mkdir -p /tmp/matplotlib /tmp/numba_cache

source /data/vision/billf/scratch/pablomer/miniconda3/etc/profile.d/conda.sh
conda activate py310-torch

export MPLCONFIGDIR=/tmp/matplotlib
export NUMBA_CACHE_DIR=/tmp/numba_cache

srun bash galaxy_images/galaxy_model/contrastive_baseline/downstream_evaluation/debugging_contrastive/run_neighbors_debug_downstream_cross_split.sh

echo "Job completed at $(date)"
