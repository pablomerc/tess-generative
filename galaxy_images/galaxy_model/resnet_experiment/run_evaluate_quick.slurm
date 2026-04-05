#!/bin/bash
#SBATCH --job-name=resnet-eval-quick
#SBATCH --qos=vision-billf-main
#SBATCH --partition=vision-billf-v100
#SBATCH --account=vision-billf
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
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

cd "${GALAXY_MODEL_DIR}"

python "${EXPERIMENT_DIR}/evaluate_generated.py" \
    --shard       "${SHARD}" \
    --checkpoint  "${CHECKPOINT}" \
    --n-galaxies  16 \
    --steps       20 \
    --no-compile
