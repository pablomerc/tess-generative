#!/bin/bash
#SBATCH --job-name=cnt-contrastive
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --partition=mi2104x
#SBATCH --output=/work1/jeroenaudenaert/pablomer/logs/contrastive_%j.out
#SBATCH --error=/work1/jeroenaudenaert/pablomer/logs/contrastive_%j.err
#SBATCH --open-mode=append

set -eo pipefail

mkdir -p /work1/jeroenaudenaert/pablomer/logs
mkdir -p /work1/jeroenaudenaert/pablomer/outputs/contrastive_baseline

# --- Environment ---
source ~/.bashrc
conda activate torchenv

# --- ROCm workaround: disable hipBLASLt (buggy on MI210 for certain matrix shapes) ---
export TORCH_BLAS_PREFER_HIPBLASLT=0

# --- Number of GPUs ---
export N_DEVICES=4

# --- Run ---
cd /work1/jeroenaudenaert/pablomer/tess-generative

export PYTHONPATH=/work1/jeroenaudenaert/pablomer/tess-generative:$PYTHONPATH

srun python -u galaxy_images/galaxy_model/contrastive_baseline/train_dual_encoder_contrastive.py

echo "Job completed at $(date)"
