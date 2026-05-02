#!/bin/bash
#SBATCH --job-name=hsc-stats-test
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:15:00
#SBATCH --partition=mi2104x
#SBATCH --output=/work1/jeroenaudenaert/pablomer/logs/hsc_stats_test_%j.out
#SBATCH --error=/work1/jeroenaudenaert/pablomer/logs/hsc_stats_test_%j.err
#SBATCH --open-mode=append

set -eo pipefail

mkdir -p /work1/jeroenaudenaert/pablomer/logs

source ~/.bashrc
conda activate torchenv

cd /work1/jeroenaudenaert/pablomer/tess-generative

srun python -u galaxy_images/galaxy_model/latent_traversal/extract_hsc_stats.py \
    --n-files 1 \
    --n-workers 1 \
    --output /work1/jeroenaudenaert/pablomer/data/hsc_ivar_psf_stats_test.h5

echo "Job completed at $(date)"
