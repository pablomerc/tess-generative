#!/bin/bash
#SBATCH --job-name=ivar-trav-full
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:30:00
#SBATCH --partition=mi2104x
#SBATCH --output=/work1/jeroenaudenaert/pablomer/logs/ivar_traversal_full_%j.out
#SBATCH --error=/work1/jeroenaudenaert/pablomer/logs/ivar_traversal_full_%j.err
#SBATCH --open-mode=append

set -eo pipefail

mkdir -p /work1/jeroenaudenaert/pablomer/logs

source ~/.bashrc
conda activate torchenv

cd /work1/jeroenaudenaert/pablomer/tess-generative

STATS=/work1/jeroenaudenaert/pablomer/data/hsc_ivar_psf_stats.h5

echo "=== [1/2] Extracting HSC stats (all 396 parquet files) ==="
srun python -u galaxy_images/galaxy_model/latent_traversal/extract_hsc_stats.py \
    --n-workers 16 \
    --output "${STATS}"

echo "=== [2/2] Running traversal analysis ==="
# Remove any stale test cache before running on the full dataset
rm -f galaxy_images/galaxy_model/latent_traversal/ivar_psf_cache.npz
srun python -u galaxy_images/galaxy_model/latent_traversal/ivar_psf_traversal_analysis.py \
    --stats-path "${STATS}" \
    --output-dir galaxy_images/galaxy_model/latent_traversal

echo "Job completed at $(date)"
