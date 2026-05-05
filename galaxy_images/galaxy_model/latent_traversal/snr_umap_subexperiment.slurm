#!/bin/bash
#SBATCH --job-name=snr-umap-subexp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:30:00
#SBATCH --partition=mi2104x
#SBATCH --exclude=k002-005
#SBATCH --output=/work1/jeroenaudenaert/pablomer/logs/snr_umap_subexp_%j.out
#SBATCH --error=/work1/jeroenaudenaert/pablomer/logs/snr_umap_subexp_%j.err
#SBATCH --open-mode=append

set -eo pipefail

mkdir -p /work1/jeroenaudenaert/pablomer/logs

source ~/.bashrc
conda activate torchenv

LATENT_DIR=/work1/jeroenaudenaert/pablomer/tess-generative/galaxy_images/galaxy_model/latent_traversal
source "$LATENT_DIR/_slurm_notify_wrapper.sh"

cd /work1/jeroenaudenaert/pablomer/tess-generative
export PYTHONPATH=/work1/jeroenaudenaert/pablomer/tess-generative:$PYTHONPATH
export TORCH_BLAS_PREFER_HIPBLASLT=0

srun python -u galaxy_images/galaxy_model/latent_traversal/snr_umap_subexperiment.py

echo "Job completed at $(date)"
