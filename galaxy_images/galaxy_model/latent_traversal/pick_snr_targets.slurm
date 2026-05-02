#!/bin/bash
#SBATCH --job-name=pick-snr-targets
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:10:00
#SBATCH --partition=mi2104x
#SBATCH --output=/work1/jeroenaudenaert/pablomer/logs/pick_snr_targets_%j.out
#SBATCH --error=/work1/jeroenaudenaert/pablomer/logs/pick_snr_targets_%j.err
#SBATCH --open-mode=append

set -eo pipefail

mkdir -p /work1/jeroenaudenaert/pablomer/logs

source ~/.bashrc
conda activate torchenv

LATENT_DIR=/work1/jeroenaudenaert/pablomer/tess-generative/galaxy_images/galaxy_model/latent_traversal
source "$LATENT_DIR/_slurm_notify_wrapper.sh"

cd /work1/jeroenaudenaert/pablomer/tess-generative

srun python -u galaxy_images/galaxy_model/latent_traversal/pick_snr_targets.py \
    --n-images 20 --p-lo 1 --p-hi 99 --n-cols 5

echo "Job completed at $(date)"
