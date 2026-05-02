#!/bin/bash
# Smoke test for the hierarchical-attention neighbors_efficient training pipeline.
# Run on an interactive session with at least 1 GPU available.
#
# Usage:
#   bash galaxy_images/galaxy_model/hierarchical_attention/smoke_neighbors_efficient_hier.sh

set -eo pipefail

source ~/.bashrc
conda activate torchenv

export TORCH_BLAS_PREFER_HIPBLASLT=0

cd /work1/jeroenaudenaert/pablomer/tess-generative
export PYTHONPATH=/work1/jeroenaudenaert/pablomer/tess-generative:$PYTHONPATH

python -u galaxy_images/galaxy_model/hierarchical_attention/neighbors_efficient_train_hier.py \
    --set trainer.num_steps=20 \
    --set trainer.devices=1 \
    --set trainer.val_check_interval=10 \
    --set trainer.checkpoint_every_n_train_steps=20 \
    --set trainer.scale_steps_by_devices=false \
    --set data.batch_size=4 \
    --set data.num_workers=2 \
    --set data.save_heldout_validation=false \
    --set wandb.enabled=false

echo "Hierarchical smoke test completed at $(date)"
