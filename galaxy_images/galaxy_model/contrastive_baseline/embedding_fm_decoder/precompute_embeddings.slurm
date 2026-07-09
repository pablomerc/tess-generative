#!/bin/bash
#SBATCH --job-name=emb-precompute
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --partition=mit_normal_gpu
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --open-mode=append
#
# Stage 1: encode the first N anchors through the frozen contrastive encoder and write the
# [embedding, image] H5. One-time. Edit the paths/vars below for your cluster.

set -eo pipefail

# ---- EDIT THESE FOR YOUR CLUSTER ----
REPO=/orcd/pool/007/pablomer/tess-generative
TORCHENV_PY=/orcd/home/002/pablomer/miniforge3/envs/torchenv/bin/python
DATA_DIR=/orcd/pool/007/pablomer/efficient_neighs
CKPT=/orcd/pool/007/pablomer/outputs/contrastive/best-epoch=599-step=240000.ckpt
OUT_H5=/orcd/pool/007/pablomer/outputs/embedding_fm/anchor_embeddings_100k.h5
LIMIT=100000
BATCH=256
# -------------------------------------

source ~/.bashrc
conda activate torchenv || true   # TORCHENV_PY is pinned below regardless

export PYTHONPATH="$REPO:$PYTHONPATH"
cd "$REPO"

echo "PYTHON=$TORCHENV_PY"
"$TORCHENV_PY" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

srun "$TORCHENV_PY" -u galaxy_images/galaxy_model/contrastive_baseline/embedding_fm_decoder/precompute_embeddings.py \
    --data-dir "$DATA_DIR" \
    --ckpt "$CKPT" \
    --out-h5 "$OUT_H5" \
    --limit "$LIMIT" \
    --batch-size "$BATCH"

echo "Precompute done at $(date)"
