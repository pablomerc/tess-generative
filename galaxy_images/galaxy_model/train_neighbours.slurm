#!/bin/bash
#SBATCH --job-name=ne-long
#SBATCH --qos=vision-billf-main
#SBATCH --partition=vision-billf-v100
#SBATCH --account=vision-billf
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=22:00:00
#SBATCH --gres=gpu:4
#SBATCH --output=/data/vision/billf/scratch/pablomer/projects/tess-generative/galaxy_images/galaxy_model/logs/job_%j.out
#SBATCH --error=/data/vision/billf/scratch/pablomer/projects/tess-generative/galaxy_images/galaxy_model/logs/job_%j.err
#SBATCH --open-mode=append

set -euo pipefail

cd /data/vision/billf/scratch/pablomer/projects/tess-generative
mkdir -p galaxy_images/galaxy_model/logs

# source ~/.bashrc
source /data/vision/billf/scratch/pablomer/miniconda3/etc/profile.d/conda.sh
conda activate py310-torch

# Load wandb key without printing it
set -a
source /data/vision/billf/scratch/pablomer/.secrets/wandb.env
set +a

srun python -u galaxy_images/galaxy_model/neighbours_train.py

echo "Job completed at $(date)"
