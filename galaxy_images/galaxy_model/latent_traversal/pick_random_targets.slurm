#!/bin/bash
#SBATCH --job-name=pick-random-targets
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:10:00
#SBATCH --partition=mi2104x
#SBATCH --output=/work1/jeroenaudenaert/pablomer/logs/pick_random_targets_%j.out
#SBATCH --error=/work1/jeroenaudenaert/pablomer/logs/pick_random_targets_%j.err
#SBATCH --open-mode=append

set -eo pipefail

mkdir -p /work1/jeroenaudenaert/pablomer/logs

source ~/.bashrc
conda activate torchenv

LATENT_DIR=/work1/jeroenaudenaert/pablomer/tess-generative/galaxy_images/galaxy_model/latent_traversal
source "$LATENT_DIR/_slurm_notify_wrapper.sh"

cd /work1/jeroenaudenaert/pablomer/tess-generative

srun python -u galaxy_images/galaxy_model/latent_traversal/pick_random_targets.py \
    --n-images 64 --n-cols 8 \
    --snr-percentile 60 --snr-half-window 2 \
    --seed 0

echo "Job completed at $(date)"
