#!/bin/bash
#SBATCH --job-name=plot-hsc-ivar-pcts
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:15:00
#SBATCH --partition=mi2104x
#SBATCH --output=/work1/jeroenaudenaert/pablomer/logs/plot_hsc_ivar_percentiles_%j.out
#SBATCH --error=/work1/jeroenaudenaert/pablomer/logs/plot_hsc_ivar_percentiles_%j.err
#SBATCH --open-mode=append

set -eo pipefail

mkdir -p /work1/jeroenaudenaert/pablomer/logs

source ~/.bashrc
conda activate torchenv

cd /work1/jeroenaudenaert/pablomer/tess-generative

srun python -u galaxy_images/galaxy_model/latent_traversal/plot_hsc_ivar_percentiles.py \
    --n-images 20 \
    --p-lo 1 \
    --p-hi 99 \
    --n-cols 5

echo "Job completed at $(date)"
