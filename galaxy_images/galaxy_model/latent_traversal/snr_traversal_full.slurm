#!/bin/bash
#SBATCH --job-name=snr-traversal-full
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:30:00
#SBATCH --partition=mi2104x
#SBATCH --output=/work1/jeroenaudenaert/pablomer/logs/snr_traversal_full_%j.out
#SBATCH --error=/work1/jeroenaudenaert/pablomer/logs/snr_traversal_full_%j.err
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

# All 6 targets: original 4 (p50–p70 picks) + 2 from rank-481 anomaly browse
TARGETS="92198,53601,91614,29493,65679,1779"
MODES="k5,k10,k15,k1x5,k1x10"

srun python -u galaxy_images/galaxy_model/latent_traversal/snr_traversal_full.py \
    --target-idxs "$TARGETS" \
    --modes "$MODES"

echo "Job completed at $(date)"
