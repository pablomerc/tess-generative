#!/bin/bash
# ==============================================================================
# Smoke test: 2 galaxies × 2 samples — verifies end-to-end pipeline quickly.
# ==============================================================================
#SBATCH --job-name=xi2-test
#SBATCH --qos=vision-billf-main
#SBATCH --partition=vision-billf-v100
#SBATCH --account=vision-billf
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1
#SBATCH --output=logs/job_%j.out
#SBATCH --error=logs/job_%j.err

set -e

# ------------------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------------------
REPO_ROOT="/data/vision/billf/scratch/pablomer/projects/tess-generative"
EXPERIMENT_DIR="${REPO_ROOT}/galaxy_images/galaxy_model/xi_squared_experiment"
GALAXY_MODEL_DIR="${REPO_ROOT}/galaxy_images/galaxy_model"

CHECKPOINT="/data/vision/billf/scratch/pablomer/projects/tess-generative/galaxy-flow-matching-neighbours/g2g9kvr4/checkpoints/latest-step=step=75000.ckpt"
SHARD="/data/vision/billf/scratch/pablomer/data/neighbor_batches/neighbors_shard_0000.h5"

OUTPUT_SAMPLES="${EXPERIMENT_DIR}/outputs/test_samples.h5"
OUTPUT_CALIB="${EXPERIMENT_DIR}/outputs/test_calibration"

# ------------------------------------------------------------------------------
# Environment
# ------------------------------------------------------------------------------
source /data/vision/billf/scratch/pablomer/miniconda3/etc/profile.d/conda.sh
conda activate py310-torch

# Run from galaxy_model/ so relative imports in the model code resolve correctly
cd "${GALAXY_MODEL_DIR}"

mkdir -p "${EXPERIMENT_DIR}/outputs"

# ------------------------------------------------------------------------------
# Step 1: Generate samples
# ------------------------------------------------------------------------------
echo "================================================================"
echo "Step 1: Generating samples (2 galaxies × 2 samples)"
echo "================================================================"
python "${EXPERIMENT_DIR}/generate_samples.py" \
    --checkpoint "${CHECKPOINT}" \
    --shard      "${SHARD}" \
    --n-galaxies 2 \
    --m-samples  2 \
    --steps      50 \
    --output     "${OUTPUT_SAMPLES}"

# ------------------------------------------------------------------------------
# Step 2: Calibration test
# ------------------------------------------------------------------------------
echo ""
echo "================================================================"
echo "Step 2: Xi-squared calibration test"
echo "================================================================"
python "${EXPERIMENT_DIR}/calibration_test.py" \
    --samples    "${OUTPUT_SAMPLES}" \
    --output-dir "${OUTPUT_CALIB}"

echo ""
echo "================================================================"
echo "Done. Outputs in ${EXPERIMENT_DIR}/outputs/"
echo "================================================================"
