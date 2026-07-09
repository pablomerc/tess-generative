#!/bin/bash
#SBATCH --job-name=emb-fm-long
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=2-00:00:00
#SBATCH --partition=pg_mki_aryeh
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --open-mode=append
#
# Stage 2 on a long-time-limit partition (pg_mki_aryeh 2 day / ou_mki_gpu 7 day) — no resume
# chaining needed. Set --partition and --time to your allocation. Multi-GPU: raise --gres and
# --devices together (DDP).

set -eo pipefail

# ---- EDIT THESE FOR YOUR CLUSTER ----
REPO=/orcd/pool/007/pablomer/tess-generative
TORCHENV_PY=/orcd/home/002/pablomer/miniforge3/envs/torchenv/bin/python
DATA_H5=/orcd/pool/007/pablomer/outputs/embedding_fm/anchor_embeddings_100k.h5
RUN_DIR=/orcd/pool/007/pablomer/outputs/embedding_fm/runs/emb-fm-adagn-concat128-long
INJECTION=adagn
COND_MODE=concat128
MAX_STEPS=200000
BATCH=128
DEVICES=1
export WANDB_API_KEY="${WANDB_API_KEY:-}"   # provide via env; do NOT commit the key
# -------------------------------------

mkdir -p "$RUN_DIR/checkpoints"

source ~/.bashrc
conda activate torchenv || true
export PYTHONPATH="$REPO:$PYTHONPATH"
export N_DEVICES="$DEVICES"
cd "$REPO"

RESUME_ARG=""
if [ -f "$RUN_DIR/checkpoints/last.ckpt" ]; then
    RESUME_ARG="--resume-from $RUN_DIR/checkpoints/last.ckpt"
fi

"$TORCHENV_PY" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

srun "$TORCHENV_PY" -u galaxy_images/galaxy_model/contrastive_baseline/embedding_fm_decoder/train_embedding_fm.py \
    --data-h5 "$DATA_H5" \
    --injection "$INJECTION" \
    --cond-mode "$COND_MODE" \
    --run-dir "$RUN_DIR" \
    --max-steps "$MAX_STEPS" \
    --batch-size "$BATCH" \
    --devices "$DEVICES" \
    $RESUME_ARG

echo "Training done at $(date)"
