#!/bin/bash
#SBATCH --job-name=snr-traversal-paper
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --partition=mi2104x
#SBATCH --output=/work1/jeroenaudenaert/pablomer/logs/snr_traversal_paper_%j.out
#SBATCH --error=/work1/jeroenaudenaert/pablomer/logs/snr_traversal_paper_%j.err
#SBATCH --open-mode=append
#SBATCH --exclude=k002-005

set -eo pipefail

mkdir -p /work1/jeroenaudenaert/pablomer/logs

source ~/.bashrc
conda activate torchenv
# Belt-and-braces: even if conda activation fails on a flaky node, fall back to the env python.
PYBIN="/work1/jeroenaudenaert/pablomer/miniconda3/envs/torchenv/bin/python"

LATENT_DIR=/work1/jeroenaudenaert/pablomer/tess-generative/galaxy_images/galaxy_model/latent_traversal
source "$LATENT_DIR/_slurm_notify_wrapper.sh"

cd /work1/jeroenaudenaert/pablomer/tess-generative
export PYTHONPATH=/work1/jeroenaudenaert/pablomer/tess-generative:$PYTHONPATH
export TORCH_BLAS_PREFER_HIPBLASLT=0

# Paper-figure target set: original favorite + 4 picks from p60 random gallery
TARGETS="53601,55427,83907,15780,33241"
MODES="k10"

OUTDIR="$LATENT_DIR/snr_traversal_paper"
mkdir -p "$OUTDIR"

srun "$PYBIN" -u galaxy_images/galaxy_model/latent_traversal/snr_traversal_full.py \
    --target-idxs "$TARGETS" \
    --modes "$MODES" \
    --output-dir "$OUTDIR" \
    --arrays-file "$OUTDIR/snr_traversal_arrays.h5" \
    --no-discord

echo "Job completed at $(date)"
