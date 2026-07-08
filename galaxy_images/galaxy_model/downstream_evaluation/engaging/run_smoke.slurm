#!/bin/bash
#SBATCH --job-name=engaging-smoke
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=0:30:00
#SBATCH --partition=mit_normal_gpu
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --open-mode=append

set -eo pipefail

SCRIPT_DIR="/home/pablomer/orcd/pool/tess-generative/galaxy_images/galaxy_model/downstream_evaluation/engaging"
REPO_ROOT="/home/pablomer/orcd/pool/tess-generative"

source ~/.bashrc
conda activate torchenv
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1
export WANDB_MODE=disabled

cd "${REPO_ROOT}"
bash "${SCRIPT_DIR}/smoke_test.sh"
