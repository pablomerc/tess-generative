#!/bin/bash
#SBATCH --job-name=disc-gen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --partition=mi2104x
#SBATCH --output=/work1/jeroenaudenaert/pablomer/logs/disc_gen_%j.out
#SBATCH --error=/work1/jeroenaudenaert/pablomer/logs/disc_gen_%j.err
#SBATCH --open-mode=append

set -eo pipefail

mkdir -p /work1/jeroenaudenaert/pablomer/logs

N_SAMPLES=${1:-1000}

# --- Environment ---
source ~/.bashrc
conda activate torchenv

# --- ROCm workaround ---
export TORCH_BLAS_PREFER_HIPBLASLT=0

# --- Run ---
cd /work1/jeroenaudenaert/pablomer/tess-generative

export PYTHONPATH=/work1/jeroenaudenaert/pablomer/tess-generative:$PYTHONPATH

echo "Generating $N_SAMPLES samples per survey"
srun python -u galaxy_images/galaxy_model/generative_evaluation/discriminator_generate_dataset.py --n-samples $N_SAMPLES

echo "Job completed at $(date)"
