#!/bin/bash
#SBATCH --job-name=xpred-full
#SBATCH --qos=vision-billf-main
#SBATCH --partition=vision-billf-v100
#SBATCH --account=vision-billf
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0-08:00:00
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

OUTDIR="galaxy_images/galaxy_model/cross_predict_experiment"

echo "=== HSC -> Legacy (50 epochs, full dataset) ==="
python -u ${OUTDIR}/cross_predict.py \
    --direction hsc_to_legacy \
    --epochs 50 \
    --output-dir ${OUTDIR}

echo ""
echo "=== Legacy -> HSC (50 epochs, full dataset) ==="
python -u ${OUTDIR}/cross_predict.py \
    --direction legacy_to_hsc \
    --epochs 50 \
    --output-dir ${OUTDIR}

echo "Full job completed at $(date)"
