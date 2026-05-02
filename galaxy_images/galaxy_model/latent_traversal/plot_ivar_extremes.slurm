#!/bin/bash
#SBATCH --job-name=plot-ivar-extremes
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --partition=mi2104x
#SBATCH --output=/work1/jeroenaudenaert/pablomer/logs/plot_ivar_extremes_%j.out
#SBATCH --error=/work1/jeroenaudenaert/pablomer/logs/plot_ivar_extremes_%j.err
#SBATCH --open-mode=append

set -eo pipefail

mkdir -p /work1/jeroenaudenaert/pablomer/logs

source ~/.bashrc
conda activate torchenv

cd /work1/jeroenaudenaert/pablomer/tess-generative

srun python -u galaxy_images/galaxy_model/latent_traversal/plot_ivar_extremes.py \
    --n-examples 200000 \
    --n-show 8

echo "Job completed at $(date)"
