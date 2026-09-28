#!/bin/bash
#SBATCH --job-name=hsc_lsst
#SBATCH --partition=mi2101x
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --time=12:00:00
#SBATCH --output=/work1/jeroenaudenaert/pablomer/logs/hsc_lsst-%j.log

# Launcher for the HSC x LSST dual-encoder run (lsst/DECISIONS.md).
#   full run:   sbatch galaxy_images/galaxy_model/lsst/hsc_lsst_train.sbatch
#   smoke test: sbatch -p devel -t 25:00 galaxy_images/galaxy_model/lsst/hsc_lsst_train.sbatch \
#                 --set trainer.num_steps=60 --set trainer.val_check_interval=30 --set wandb.enabled=false
# Extra arguments are passed straight to the trainer as --set overrides.
# Do NOT add --mem: the submission filter on this cluster rejects it.

export TORCH_BLAS_PREFER_HIPBLASLT=0                 # MI210 hipBLASLt bug (env half; python half is in hsc_lsst_train.py)
# Writable per-job MIOpen cache; without it MIOpen can die with miopenStatusInternalError.
export MIOPEN_USER_DB_PATH=/tmp/miopen_${USER}_${SLURM_JOB_ID}
export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_USER_DB_PATH
mkdir -p "$MIOPEN_USER_DB_PATH" /tmp/mpl_${USER} /tmp/numba_${USER}
export MPLCONFIGDIR=/tmp/mpl_${USER}
export NUMBA_CACHE_DIR=/tmp/numba_${USER}
export WANDB_ENTITY=tess-ml                         # project lsst-disentanglement lives under the team entity

source /work1/jeroenaudenaert/pablomer/miniconda3/etc/profile.d/conda.sh
conda activate torchenv

cd /work1/jeroenaudenaert/pablomer/tess-generative
export PYTHONPATH=.
echo "[launcher] node=$(hostname) job=$SLURM_JOB_ID partition=$SLURM_JOB_PARTITION args: $*"
python -m galaxy_images.galaxy_model.lsst.hsc_lsst_train "$@"
