#!/bin/bash
#SBATCH --job-name=aion-umap-ne
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --time=06:00:00
#SBATCH --partition=mi2104x
#SBATCH --output=/work1/jeroenaudenaert/pablomer/logs/aion_umap_ne_%j.out
#SBATCH --error=/work1/jeroenaudenaert/pablomer/logs/aion_umap_ne_%j.err
#SBATCH --open-mode=append

set -eo pipefail

mkdir -p /work1/jeroenaudenaert/pablomer/logs

# --- Environment ---
source ~/.bashrc
conda activate torchenv

# --- ROCm workaround (no GPU used here, but kept for parity) ---
export TORCH_BLAS_PREFER_HIPBLASLT=0

# --- Run ---
cd /work1/jeroenaudenaert/pablomer/tess-generative
export PYTHONPATH=/work1/jeroenaudenaert/pablomer/tess-generative:$PYTHONPATH

: "${NUM_ANCHORS:=4096}"
: "${BATCH_SIZE:=32}"

srun python -u galaxy_images/galaxy_model/aion_benchmark/aion_umap/aion_umap_neighbors_efficient.py \
    --data-dir    /work1/jeroenaudenaert/pablomer/data/neighbors_efficient \
    --num-anchors "$NUM_ANCHORS" \
    --batch-size  "$BATCH_SIZE" \
    --webhook     "$DISCORD_WEBHOOK"

echo "Job completed at $(date)"
