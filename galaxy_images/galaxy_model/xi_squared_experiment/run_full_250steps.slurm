#!/bin/bash
# ==============================================================================
# Full experiment: 100 galaxies × 20 samples at 250 integration steps.
# This is the publication-quality version; run_full.slurm uses 50 steps.
# ==============================================================================
#SBATCH --job-name=xi2-250
#SBATCH --qos=vision-billf-main
#SBATCH --partition=vision-billf-v100
#SBATCH --account=vision-billf
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
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

OUTPUT_SAMPLES="${EXPERIMENT_DIR}/outputs_250steps/full_samples_250steps.h5"
OUTPUT_CALIB="${EXPERIMENT_DIR}/outputs_250steps/full_calibration"

# ------------------------------------------------------------------------------
# Environment
# ------------------------------------------------------------------------------
source /data/vision/billf/scratch/pablomer/miniconda3/etc/profile.d/conda.sh
conda activate py310-torch

cd "${GALAXY_MODEL_DIR}"

mkdir -p "${EXPERIMENT_DIR}/outputs_250steps"

# ------------------------------------------------------------------------------
# Step 1: Generate samples
# ------------------------------------------------------------------------------
echo "================================================================"
echo "Step 1: Generating samples (100 galaxies × 20 samples × 250 steps)"
echo "================================================================"
python "${EXPERIMENT_DIR}/generate_samples.py" \
    --checkpoint "${CHECKPOINT}" \
    --shard      "${SHARD}" \
    --n-galaxies 100 \
    --m-samples  20 \
    --steps      250 \
    --output     "${OUTPUT_SAMPLES}"

# ------------------------------------------------------------------------------
# Step 2: Calibration test
# ------------------------------------------------------------------------------
echo ""
echo "================================================================"
echo "Step 2: Xi-squared calibration test (split by anchor survey)"
echo "================================================================"
python "${EXPERIMENT_DIR}/calibration_test.py" \
    --samples    "${OUTPUT_SAMPLES}" \
    --output-dir "${OUTPUT_CALIB}"

echo ""
echo "================================================================"
echo "Done. Results:"
echo "  Samples  : ${OUTPUT_SAMPLES}"
echo "  Figures  : ${OUTPUT_CALIB}/calibration_{hsc,legacy}.png"
echo "  Reports  : ${OUTPUT_CALIB}/calibration_report_{hsc,legacy}.txt"
echo "  JSON     : ${OUTPUT_CALIB}/calibration_stats_{hsc,legacy}.json"
echo "================================================================"
