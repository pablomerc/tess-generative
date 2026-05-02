#!/bin/bash
#SBATCH --job-name=ivar-trav-test
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:20:00
#SBATCH --partition=mi2104x
#SBATCH --output=/work1/jeroenaudenaert/pablomer/logs/ivar_traversal_test_%j.out
#SBATCH --error=/work1/jeroenaudenaert/pablomer/logs/ivar_traversal_test_%j.err
#SBATCH --open-mode=append

set -eo pipefail

mkdir -p /work1/jeroenaudenaert/pablomer/logs

source ~/.bashrc
conda activate torchenv

cd /work1/jeroenaudenaert/pablomer/tess-generative

STATS=/work1/jeroenaudenaert/pablomer/data/hsc_ivar_psf_stats_test.h5

echo "=== [1/2] Extracting HSC stats (1 parquet file) ==="
srun python -u galaxy_images/galaxy_model/latent_traversal/extract_hsc_stats.py \
    --n-files 1 \
    --n-workers 1 \
    --output "${STATS}"

echo "=== [2/2] Running traversal analysis ==="
srun python -u galaxy_images/galaxy_model/latent_traversal/ivar_psf_traversal_analysis.py \
    --stats-path "${STATS}" \
    --output-dir galaxy_images/galaxy_model/latent_traversal

echo "Job completed at $(date)"
