#!/bin/bash
#SBATCH --job-name=ne-hier-global-ins
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --partition=mi2104x
#SBATCH --output=/work1/jeroenaudenaert/pablomer/logs/hier_global_ins_%j.out
#SBATCH --error=/work1/jeroenaudenaert/pablomer/logs/hier_global_ins_%j.err
#SBATCH --open-mode=append

set -eo pipefail

mkdir -p /work1/jeroenaudenaert/pablomer/logs

# --- Environment ---
source ~/.bashrc
conda activate torchenv

# --- Weights & Biases ---
# Reuse credentials from a prior `wandb login`. If WANDB_API_KEY is set in the
# submit shell, W&B will use it automatically.

# --- ROCm workaround: disable hipBLASLt (buggy on MI210 for certain matrix shapes) ---
export TORCH_BLAS_PREFER_HIPBLASLT=0

# --- Run ---
cd /work1/jeroenaudenaert/pablomer/tess-generative

export PYTHONPATH=/work1/jeroenaudenaert/pablomer/tess-generative:$PYTHONPATH

srun python -u galaxy_images/galaxy_model/hierarchical_attention/neighbours_train_hier_global_ins.py

echo "Job completed at $(date)"
