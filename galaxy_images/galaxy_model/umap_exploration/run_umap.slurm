#!/bin/bash
#SBATCH --job-name=umap-neigh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --partition=mi2104x
#SBATCH --output=/work1/jeroenaudenaert/pablomer/logs/umap_neigh_%j.out
#SBATCH --error=/work1/jeroenaudenaert/pablomer/logs/umap_neigh_%j.err
#SBATCH --open-mode=append

set -eo pipefail

mkdir -p /work1/jeroenaudenaert/pablomer/logs

N_EXAMPLES=${1:-2000}

# --- Environment ---
source ~/.bashrc
conda activate torchenv

# --- ROCm workaround ---
export TORCH_BLAS_PREFER_HIPBLASLT=0

# --- Run ---
cd /work1/jeroenaudenaert/pablomer/tess-generative

export PYTHONPATH=/work1/jeroenaudenaert/pablomer/tess-generative:$PYTHONPATH

echo "Running UMAP neighbors exploration with $N_EXAMPLES examples"
srun python -u galaxy_images/galaxy_model/umap_exploration/umap_neighbors_exploration.py --n-examples $N_EXAMPLES

echo "Job completed at $(date)"
