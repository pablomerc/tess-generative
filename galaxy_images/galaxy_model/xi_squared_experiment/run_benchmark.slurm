#!/bin/bash
# ==============================================================================
# Compile benchmark: measures torch.compile speedup for flow matching inference.
# ==============================================================================
#SBATCH --job-name=xi2-bench
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

# ------------------------------------------------------------------------------
# Environment
# ------------------------------------------------------------------------------
source /data/vision/billf/scratch/pablomer/miniconda3/etc/profile.d/conda.sh
conda activate py310-torch

cd "${GALAXY_MODEL_DIR}"

mkdir -p "${EXPERIMENT_DIR}/logs"

# ------------------------------------------------------------------------------
# Run benchmark
# ------------------------------------------------------------------------------
echo "================================================================"
echo "torch.compile speedup benchmark"
echo "================================================================"
python "${EXPERIMENT_DIR}/compile_benchmark.py" \
    --checkpoint "${CHECKPOINT}" \
    --shard      "${SHARD}" \
    --n-warmup   2 \
    --n-trials   10 \
    --steps      50

echo ""
echo "Done. Results are printed above — check logs/job_${SLURM_JOB_ID}.out"
