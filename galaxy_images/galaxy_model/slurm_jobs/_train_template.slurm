#!/bin/bash
# Generic SLURM template for galaxy_model training. Each per-job script
# defines JOB_NAME and CONFIG_PATH and then sources or invokes this body.
#
# Required environment variables:
#   JOB_NAME    — short job tag (used for SBATCH --job-name and log filenames)
#   CONFIG_PATH — absolute path to the experiment config JSON

set -eo pipefail

mkdir -p /work1/jeroenaudenaert/pablomer/logs

source ~/.bashrc
conda activate torchenv

# Bypass `conda activate` flakiness inside srun by pinning the env's python
# directly (the activate-on-login worked, but DDP child shells sometimes
# don't inherit the env on this cluster).
TORCHENV_PY=/work1/jeroenaudenaert/pablomer/miniconda3/envs/torchenv/bin/python

# ROCm workaround
export TORCH_BLAS_PREFER_HIPBLASLT=0

# WandB
export WANDB_API_KEY="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"

cd /work1/jeroenaudenaert/pablomer/tess-generative
export PYTHONPATH=/work1/jeroenaudenaert/pablomer/tess-generative:$PYTHONPATH

echo "JOB_NAME=$JOB_NAME"
echo "CONFIG_PATH=$CONFIG_PATH"
echo "PYTHON=$TORCHENV_PY"
"$TORCHENV_PY" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

srun "$TORCHENV_PY" -u galaxy_images/galaxy_model/generic_train.py --config "$CONFIG_PATH"

echo "Job completed at $(date)"
