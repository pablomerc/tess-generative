#!/bin/bash
#SBATCH --job-name=ode-eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --partition=mi2104x
#SBATCH --output=/work1/jeroenaudenaert/pablomer/logs/ode_eval_%j.out
#SBATCH --error=/work1/jeroenaudenaert/pablomer/logs/ode_eval_%j.err
#SBATCH --open-mode=append

set -eo pipefail

mkdir -p /work1/jeroenaudenaert/pablomer/logs

# --- Environment ---
source ~/.bashrc
conda activate torchenv

# --- ROCm workaround: disable hipBLASLt (buggy on MI210 for certain matrix shapes) ---
export TORCH_BLAS_PREFER_HIPBLASLT=0

# --- Run ---
cd /work1/jeroenaudenaert/pablomer/tess-generative

export PYTHONPATH=/work1/jeroenaudenaert/pablomer/tess-generative:$PYTHONPATH

srun python -u galaxy_images/galaxy_model/generative_evaluation/ode_progression_analysis.py

echo "Job completed at $(date)"
