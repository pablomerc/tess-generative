#!/bin/bash
#SBATCH -J refit-arm
#SBATCH -o /home/pablomer/orcd/pool/tess-generative/galaxy_images/galaxy_model/contrastive_anomalies/logs/arm.%j.out
#SBATCH -e /home/pablomer/orcd/pool/tess-generative/galaxy_images/galaxy_model/contrastive_anomalies/logs/arm.%j.err
#SBATCH -t 02:00:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH --mem=48G

# One arm (both std and raw variants) per job, so all six arms run concurrently
# across partitions instead of queueing behind the single-GPU QOS on
# mit_normal_gpu. Pass ARM and POOL through --export.

set -o pipefail

source /orcd/home/002/pablomer/miniforge3/etc/profile.d/conda.sh
conda activate torchenv

WORKDIR="/orcd/pool/007/pablomer/tess-generative/galaxy_images/galaxy_model"
cd "$WORKDIR"
POOL="${POOL:-367k}"
echo "[arm] ARM=$ARM POOL=$POOL host=$(hostname -s) part=$SLURM_JOB_PARTITION"
nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1

python -u contrastive_anomalies/refit_flows_474k.py \
    --pool "$POOL" --variants std raw --arms "$ARM"
