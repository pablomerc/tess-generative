#!/bin/bash
#SBATCH --job-name=xpred-test
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

set -a
source /data/vision/billf/scratch/pablomer/.secrets/wandb.env
set +a

echo "=== Quick test: HSC -> Legacy (3 epochs, 1024 samples) ==="
python -u galaxy_images/galaxy_model/cross_predict_experiment/cross_predict.py \
    --direction hsc_to_legacy \
    --epochs 3 \
    --quick-test \
    --no-discord \
    --output-dir galaxy_images/galaxy_model/cross_predict_experiment

echo ""
echo "=== Quick test: Legacy -> HSC (3 epochs, 1024 samples) ==="
python -u galaxy_images/galaxy_model/cross_predict_experiment/cross_predict.py \
    --direction legacy_to_hsc \
    --epochs 3 \
    --quick-test \
    --no-discord \
    --output-dir galaxy_images/galaxy_model/cross_predict_experiment

echo "Test job completed at $(date)"
