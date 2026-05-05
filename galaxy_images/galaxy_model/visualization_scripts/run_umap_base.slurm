#!/bin/bash
#SBATCH --job-name=umap-base
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --partition=mi2104x
#SBATCH --output=/work1/jeroenaudenaert/pablomer/logs/umap_base_%x_%j.out
#SBATCH --error=/work1/jeroenaudenaert/pablomer/logs/umap_base_%x_%j.err
#SBATCH --open-mode=append

set -eo pipefail

mkdir -p /work1/jeroenaudenaert/pablomer/logs

# --- Required env vars (set via --export at submit time) ---
: "${NUM_ANCHORS:?NUM_ANCHORS env var is required}"
: "${TAG:?TAG env var is required}"
: "${UMAP_SEED:=42}"

# --- Environment ---
source ~/.bashrc
conda activate torchenv

# --- ROCm workaround (HipBLASLt is buggy on MI210 for some shapes) ---
export TORCH_BLAS_PREFER_HIPBLASLT=0

# --- Run from repo root so module imports resolve ---
cd /work1/jeroenaudenaert/pablomer/tess-generative
export PYTHONPATH=/work1/jeroenaudenaert/pablomer/tess-generative:$PYTHONPATH

srun python -u galaxy_images/galaxy_model/visualization_scripts/regenerate_umap_base.py \
    --checkpoint  galaxy_images/galaxy_model/checkpoints/base/snapshot.ckpt \
    --data-dir    /work1/jeroenaudenaert/pablomer/data/neighbors_efficient \
    --num-anchors "$NUM_ANCHORS" \
    --tag         "$TAG" \
    --umap-seed   "$UMAP_SEED" \
    --epoch       87000 \
    --webhook     "$DISCORD_WEBHOOK"

echo "Job completed at $(date)"
