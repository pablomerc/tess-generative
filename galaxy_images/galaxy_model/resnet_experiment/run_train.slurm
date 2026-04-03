#!/bin/bash
# ==============================================================================
# Train ResNet-18 to predict galaxy ellipticity (SHAPE_E1, SHAPE_E2).
# Reads resnet_data.h5 (produced by run_prepare_data.slurm).
# Output: resnet_best.pth, split_indices.json, outputs/*.png
# ==============================================================================
#SBATCH --job-name=resnet-train
#SBATCH --qos=vision-billf-main
#SBATCH --partition=vision-billf-v100
#SBATCH --account=vision-billf
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=logs/job_%j.out
#SBATCH --error=logs/job_%j.err

set -e

REPO_ROOT="/data/vision/billf/scratch/pablomer/projects/tess-generative"
EXPERIMENT_DIR="${REPO_ROOT}/galaxy_images/galaxy_model/resnet_experiment"

source /data/vision/billf/scratch/pablomer/miniconda3/etc/profile.d/conda.sh
conda activate py310-torch

cd "${EXPERIMENT_DIR}"

echo "================================================================"
echo "Training ResNet-18 ellipticity predictor"
echo "================================================================"
python train.py \
    --epochs      30 \
    --batch-size  512 \
    --lr          1e-3 \
    --num-workers 0

echo ""
echo "================================================================"
echo "Done."
echo "  Checkpoint : ${EXPERIMENT_DIR}/resnet_best.pth"
echo "  Split      : ${EXPERIMENT_DIR}/split_indices.json"
echo "  Plots      : ${EXPERIMENT_DIR}/outputs/"
echo "================================================================"

DISCORD_WEBHOOK="https://discord.com/api/webhooks/1488692651334177071/8b8KvACfQIVYCNY3ovee04BixCEWiqbqp1iQk4z9sXHlgR29kMkGIjl1pahV5uEPSbxe"
curl -s -F "content=✅ **Job \`${SLURM_JOB_ID}\` done**: resnet-train — checkpoint saved" \
    "${DISCORD_WEBHOOK}" || true

for FIG in \
    "${EXPERIMENT_DIR}/outputs/train_val_loss.png" \
    "${EXPERIMENT_DIR}/outputs/test_scatter_e1.png" \
    "${EXPERIMENT_DIR}/outputs/test_scatter_e2.png"; do
    if [ -f "${FIG}" ]; then
        curl -s -F "file=@${FIG}" "${DISCORD_WEBHOOK}" || true
    fi
done
