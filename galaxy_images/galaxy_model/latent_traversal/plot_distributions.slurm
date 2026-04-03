#!/bin/bash
#SBATCH --job-name=plot-dist
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --partition=mi2104x
#SBATCH --output=/work1/jeroenaudenaert/pablomer/logs/plot_dist_%j.out
#SBATCH --error=/work1/jeroenaudenaert/pablomer/logs/plot_dist_%j.err
#SBATCH --open-mode=append

set -eo pipefail

mkdir -p /work1/jeroenaudenaert/pablomer/logs

# --- Environment ---
source ~/.bashrc
conda activate torchenv

# --- Run ---
cd /work1/jeroenaudenaert/pablomer/tess-generative

srun python -u galaxy_images/galaxy_model/latent_traversal/plot_distributions.py --n-examples 8000

echo "Job completed at $(date)"
