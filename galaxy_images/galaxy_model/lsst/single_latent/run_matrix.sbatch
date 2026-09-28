#!/bin/bash
#SBATCH --job-name=lsst_matrix
#SBATCH --partition=mi2101x
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --time=12:00:00
#SBATCH --output=/work1/jeroenaudenaert/pablomer/logs/lsst_matrix-%j.log

# Generic launcher for the LSST experiment matrix:
#   sbatch run_matrix.sbatch <python -m module> [args...]
# e.g. sbatch run_matrix.sbatch galaxy_images.galaxy_model.lsst.single_latent.fm_ae.train --field cosmos
# Override partition with `sbatch -p devel -t 25:00 ...` for smoke tests.
export TORCH_BLAS_PREFER_HIPBLASLT=0                 # MI210 hipBLASLt bug
# Writable per-job MIOpen cache — without it MIOpen can die with
# miopenStatusInternalError on unusual conv shapes (seen on MI210 + MI300X).
export MIOPEN_USER_DB_PATH=/tmp/miopen_${USER}_${SLURM_JOB_ID}
export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_USER_DB_PATH
mkdir -p "$MIOPEN_USER_DB_PATH"
export DISCORD_WEBHOOK_URL=$(cat ~/.config/discord_webhook)

source /work1/jeroenaudenaert/pablomer/miniconda3/etc/profile.d/conda.sh
conda activate torchenv

cd /work1/jeroenaudenaert/pablomer/tess-generative
export PYTHONPATH=.
MODULE=$1; shift
python -m "$MODULE" "$@"
