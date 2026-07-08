#!/bin/bash
#SBATCH --job-name=recon-base
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:30:00
#SBATCH --partition=mi2104x
#SBATCH --output=/work1/jeroenaudenaert/pablomer/logs/recon_base_%x_%j.out
#SBATCH --error=/work1/jeroenaudenaert/pablomer/logs/recon_base_%x_%j.err
#SBATCH --open-mode=append

set -eo pipefail

mkdir -p /work1/jeroenaudenaert/pablomer/logs

# --- Required env vars (set via --export at submit time) ---
: "${TAG:?TAG env var is required}"
: "${NUM_EXAMPLES:=16}"
: "${NUM_SAMPLES:=5}"
: "${SEED:=42}"
: "${BOTH_DIRECTIONS:=0}"

BOTH_FLAG=""
if [[ "$BOTH_DIRECTIONS" == "1" ]]; then
    BOTH_FLAG="--both-directions"
fi

# --- Environment ---
source ~/.bashrc
conda activate torchenv

# --- ROCm workaround (HipBLASLt is buggy on MI210 for some shapes) ---
export TORCH_BLAS_PREFER_HIPBLASLT=0

# --- Run from repo root so module imports resolve ---
cd /work1/jeroenaudenaert/pablomer/tess-generative
export PYTHONPATH=/work1/jeroenaudenaert/pablomer/tess-generative:$PYTHONPATH

srun python -u galaxy_images/galaxy_model/visualization_scripts/for_paper/reconstruction_base.py \
    --checkpoint   galaxy_images/galaxy_model/checkpoints/base/snapshot.ckpt \
    --data-dir     /work1/jeroenaudenaert/pablomer/data/neighbors_efficient \
    --num-examples "$NUM_EXAMPLES" \
    --num-samples  "$NUM_SAMPLES" \
    --seed         "$SEED" \
    --tag          "$TAG" \
    --webhook      "$DISCORD_WEBHOOK" \
    $BOTH_FLAG

echo "Job completed at $(date)"
