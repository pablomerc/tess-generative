#!/bin/bash
#SBATCH --job-name=hsc-ivar-maps
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:30:00
#SBATCH --partition=mi2104x
#SBATCH --output=/work1/jeroenaudenaert/pablomer/logs/hsc_ivar_maps_%j.out
#SBATCH --error=/work1/jeroenaudenaert/pablomer/logs/hsc_ivar_maps_%j.err
#SBATCH --open-mode=append

set -eo pipefail

mkdir -p /work1/jeroenaudenaert/pablomer/logs

source ~/.bashrc
conda activate torchenv

LATENT_DIR=/work1/jeroenaudenaert/pablomer/tess-generative/galaxy_images/galaxy_model/latent_traversal
source "$LATENT_DIR/_slurm_notify_wrapper.sh"

cd /work1/jeroenaudenaert/pablomer/tess-generative

srun python -u galaxy_images/galaxy_model/latent_traversal/extract_hsc_ivar_maps.py \
    --n-workers 16 \
    --output /work1/jeroenaudenaert/pablomer/data/hsc_ivar_maps.h5

echo "Job completed at $(date)"
