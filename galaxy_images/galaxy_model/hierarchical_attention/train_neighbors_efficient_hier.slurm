#!/bin/bash
#SBATCH --job-name=ne-efficient-hier
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --partition=mi2104x
#SBATCH --output=/work1/jeroenaudenaert/pablomer/logs/train_neighbors_efficient_hier_%j.out
#SBATCH --error=/work1/jeroenaudenaert/pablomer/logs/train_neighbors_efficient_hier_%j.err
#SBATCH --open-mode=append

set -eo pipefail

mkdir -p /work1/jeroenaudenaert/pablomer/logs

# --- Environment ---
source ~/.bashrc
conda activate torchenv

# --- ROCm workaround: disable hipBLASLt (buggy on MI210 for certain matrix shapes) ---
export TORCH_BLAS_PREFER_HIPBLASLT=0

# --- WandB ---
export WANDB_API_KEY="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"

# --- Run ---
cd /work1/jeroenaudenaert/pablomer/tess-generative

export PYTHONPATH=/work1/jeroenaudenaert/pablomer/tess-generative:$PYTHONPATH

srun python -u galaxy_images/galaxy_model/hierarchical_attention/neighbors_efficient_train_hier.py

echo "Job completed at $(date)"
