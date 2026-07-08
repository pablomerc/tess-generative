#!/bin/bash
# Sanity-check the interactive compute-node env before running the smoke test.
set +e

echo "=== host / gpu ==="
hostname
nvidia-smi -L

echo ""
echo "=== conda / python ==="
source ~/.bashrc
conda activate torchenv
which python
python -c "import torch; print('torch', torch.__version__, 'cuda?', torch.cuda.is_available(), 'ndev', torch.cuda.device_count())"
python -c "import wandb; print('wandb', wandb.__version__)"

echo ""
echo "=== data + checkpoint paths ==="
for p in \
    /home/pablomer/orcd/scratch/hsc_downstream/catalog.parquet \
    /home/pablomer/orcd/scratch/legacy_downstream_full/full_1M/catalog.parquet \
    /home/pablomer/orcd/pool/tess-generative/galaxy_images/galaxy_model/checkpoints/base/snapshot.ckpt; do
    if [[ -e "$p" ]]; then
        echo "OK   $p"
    else
        echo "MISS $p"
    fi
done
