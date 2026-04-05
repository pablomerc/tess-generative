#!/bin/bash
# ==============================================================================
# Evaluate trained ResNet on flow-matching generated images.
# Generates 1 image per HSC galaxy from a neighbors shard, runs ResNet on
# real and generated images, and compares ellipticity predictions.
# ==============================================================================
#SBATCH --job-name=resnet-eval-gen
#SBATCH --qos=vision-billf-main
#SBATCH --partition=vision-billf-v100
#SBATCH --account=vision-billf
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=logs/job_%j.out
#SBATCH --error=logs/job_%j.err

set -e

REPO_ROOT="/data/vision/billf/scratch/pablomer/projects/tess-generative"
EXPERIMENT_DIR="${REPO_ROOT}/galaxy_images/galaxy_model/resnet_experiment"
GALAXY_MODEL_DIR="${REPO_ROOT}/galaxy_images/galaxy_model"

CHECKPOINT="${REPO_ROOT}/galaxy-flow-matching-neighbours/g2g9kvr4/checkpoints/latest-step=step=75000.ckpt"
SHARD="/data/vision/billf/scratch/pablomer/data/neighbor_batches/neighbors_shard_0000.h5"

source /data/vision/billf/scratch/pablomer/miniconda3/etc/profile.d/conda.sh
conda activate py310-torch

# Run from galaxy_model/ so double_train_fm_neighbors imports resolve
cd "${GALAXY_MODEL_DIR}"

echo "================================================================"
echo "Evaluating ResNet on generated images"
echo "  Shard      : ${SHARD}"
echo "  Checkpoint : ${CHECKPOINT}"
echo "================================================================"
python "${EXPERIMENT_DIR}/evaluate_generated.py" \
    --shard       "${SHARD}" \
    --checkpoint  "${CHECKPOINT}" \
    --n-galaxies  200 \
    --steps       250

echo ""
echo "================================================================"
echo "Done. Outputs: ${EXPERIMENT_DIR}/outputs/"
echo "================================================================"

DISCORD_WEBHOOK="https://discord.com/api/webhooks/1488692651334177071/8b8KvACfQIVYCNY3ovee04BixCEWiqbqp1iQk4z9sXHlgR29kMkGIjl1pahV5uEPSbxe"
curl -s -F "content=✅ **Job \`${SLURM_JOB_ID}\` done**: resnet-eval-gen — ellipticity evaluation on generated images" \
    "${DISCORD_WEBHOOK}" || true

for FIG in \
    "${EXPERIMENT_DIR}/outputs/example_images.png" \
    "${EXPERIMENT_DIR}/outputs/generated_vs_real_e1.png" \
    "${EXPERIMENT_DIR}/outputs/generated_vs_real_e2.png" \
    "${EXPERIMENT_DIR}/outputs/residuals_e1.png" \
    "${EXPERIMENT_DIR}/outputs/residuals_e2.png"; do
    if [ -f "${FIG}" ]; then
        curl -s -F "file=@${FIG}" "${DISCORD_WEBHOOK}" || true
    fi
done
