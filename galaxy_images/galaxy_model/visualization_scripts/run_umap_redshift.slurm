#!/bin/bash
#SBATCH --job-name=umap-z
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --partition=mi2104x
#SBATCH --output=/work1/jeroenaudenaert/pablomer/logs/umap_redshift_%x_%j.out
#SBATCH --error=/work1/jeroenaudenaert/pablomer/logs/umap_redshift_%x_%j.err
#SBATCH --open-mode=append

set -eo pipefail
mkdir -p /work1/jeroenaudenaert/pablomer/logs

source ~/.bashrc
conda activate torchenv
export TORCH_BLAS_PREFER_HIPBLASLT=0

cd /work1/jeroenaudenaert/pablomer/tess-generative
export PYTHONPATH=/work1/jeroenaudenaert/pablomer/tess-generative:$PYTHONPATH

: "${TAG:=overlap_redshift}"
: "${UMAP_SEED:=314}"
: "${TOP_K:=16}"

# 1) Encode + UMAP + 2x2 plot (survey + redshift coloring) — pushed to Discord
srun python -u galaxy_images/galaxy_model/visualization_scripts/regenerate_umap_redshift.py \
    --tag       "$TAG" \
    --umap-seed "$UMAP_SEED" \
    --webhook   "$DISCORD_WEBHOOK"

# 2) Suspect overlay (colored by survey) + redshift histogram — pushed to Discord
srun python -u galaxy_images/galaxy_model/visualization_scripts/find_redshift_anomalies.py \
    --tag    "$TAG" \
    --top-k  "$TOP_K" \
    --webhook "$DISCORD_WEBHOOK"

echo "Job completed at $(date)"
