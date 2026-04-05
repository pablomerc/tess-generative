#!/bin/bash
#SBATCH --job-name=resnet-eval-1024g250s
#SBATCH --qos=vision-billf-main
#SBATCH --partition=vision-billf-v100
#SBATCH --account=vision-billf
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=logs/job_%j.out
#SBATCH --error=logs/job_%j.err

set -e

REPO_ROOT="/data/vision/billf/scratch/pablomer/projects/tess-generative"
EXPERIMENT_DIR="${REPO_ROOT}/galaxy_images/galaxy_model/resnet_experiment"
GALAXY_MODEL_DIR="${REPO_ROOT}/galaxy_images/galaxy_model"

CHECKPOINT="${REPO_ROOT}/galaxy-flow-matching-neighbours/g2g9kvr4/checkpoints/latest-step=step=75000.ckpt"
SHARD="/data/vision/billf/scratch/pablomer/data/neighbor_batches/neighbors_shard_0000.h5"
OUTPUT_DIR="${EXPERIMENT_DIR}/outputs_1024g_250steps"

source /data/vision/billf/scratch/pablomer/miniconda3/etc/profile.d/conda.sh
conda activate py310-torch

cd "${GALAXY_MODEL_DIR}"

python "${EXPERIMENT_DIR}/evaluate_generated.py" \
    --shard       "${SHARD}" \
    --checkpoint  "${CHECKPOINT}" \
    --n-galaxies  1024 \
    --steps       250 \
    --output-dir  "${OUTPUT_DIR}"

DISCORD_WEBHOOK="https://discord.com/api/webhooks/1488692651334177071/8b8KvACfQIVYCNY3ovee04BixCEWiqbqp1iQk4z9sXHlgR29kMkGIjl1pahV5uEPSbxe"
curl -s -F "content=**Job \`${SLURM_JOB_ID}\` done**: resnet-eval 1024 galaxies, 250 steps" \
    "${DISCORD_WEBHOOK}" || true

for FIG in "${OUTPUT_DIR}"/*.png; do
    [ -f "${FIG}" ] && curl -s -F "file=@${FIG}" "${DISCORD_WEBHOOK}" || true
done
