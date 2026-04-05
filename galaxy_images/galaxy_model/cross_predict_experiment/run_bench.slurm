#!/bin/bash
#SBATCH --job-name=xpred-bench
#SBATCH --qos=vision-billf-main
#SBATCH --partition=vision-billf-v100
#SBATCH --account=vision-billf
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=0-00:15:00
#SBATCH --gres=gpu:1
#SBATCH --output=logs/job_%j.out
#SBATCH --error=logs/job_%j.err

set -euo pipefail

cd /data/vision/billf/scratch/pablomer/projects/tess-generative

source /data/vision/billf/scratch/pablomer/miniconda3/etc/profile.d/conda.sh
conda activate py310-torch

python -u galaxy_images/galaxy_model/cross_predict_experiment/bench_dataloader.py
