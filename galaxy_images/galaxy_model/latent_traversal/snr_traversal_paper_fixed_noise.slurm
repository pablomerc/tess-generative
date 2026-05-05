#!/bin/bash
#SBATCH --job-name=snr-paper-fixed
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --partition=mi2104x
#SBATCH --output=/work1/jeroenaudenaert/pablomer/logs/snr_traversal_paper_fixed_%j.out
#SBATCH --error=/work1/jeroenaudenaert/pablomer/logs/snr_traversal_paper_fixed_%j.err
#SBATCH --open-mode=append
#SBATCH --exclude=k002-005

set -eo pipefail

mkdir -p /work1/jeroenaudenaert/pablomer/logs

source ~/.bashrc
conda activate torchenv
PYBIN="/work1/jeroenaudenaert/pablomer/miniconda3/envs/torchenv/bin/python"

LATENT_DIR=/work1/jeroenaudenaert/pablomer/tess-generative/galaxy_images/galaxy_model/latent_traversal
source "$LATENT_DIR/_slurm_notify_wrapper.sh"

cd /work1/jeroenaudenaert/pablomer/tess-generative
export PYTHONPATH=/work1/jeroenaudenaert/pablomer/tess-generative:$PYTHONPATH
export TORCH_BLAS_PREFER_HIPBLASLT=0

# Original 5 paper targets + 4 AION anomalies (ranks #25, #28, #76, #337)
# + 1 user-selected from p60 gallery (92028):
#   #25 -> stats_idx=58095  (hdf5_row=181483)
#   #28 -> stats_idx=34953  (hdf5_row=146628)
#   #76 -> stats_idx=62309  (hdf5_row=185809)
#  #337 -> stats_idx=62218  (hdf5_row=185717)
TARGETS="53601,55427,83907,15780,33241,58095,34953,62309,62218,92028"
MODES="k10"

OUTDIR="$LATENT_DIR/snr_traversal_paper_fixed_noise"
mkdir -p "$OUTDIR"

srun "$PYBIN" -u galaxy_images/galaxy_model/latent_traversal/snr_traversal_full.py \
    --target-idxs "$TARGETS" \
    --modes "$MODES" \
    --output-dir "$OUTDIR" \
    --arrays-file "$OUTDIR/snr_traversal_arrays.h5" \
    --fixed-noise \
    --no-discord

echo "Job completed at $(date)"
