#!/bin/bash
#SBATCH --job-name=emb-fm-6h
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --partition=mit_normal_gpu
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --open-mode=append
#
# Stage 2 on a 6 h-capped partition (mit_normal_gpu) with AUTO-RESUBMIT until max-steps.
#
# How resume works:
#   - periodic ModelCheckpoint(save_last=True) writes $RUN_DIR/checkpoints/last.ckpt
#     (model + optimizer + scheduler + global_step).
#   - A successor job is queued with --dependency=afterany BEFORE training starts, so even a
#     SLURM timeout kill is followed by a resume. Each job checks $RUN_DIR/DONE first and exits
#     if training already reached max-steps. A resubmit counter caps runaway chains.
#   - Submit ONCE: sbatch train_embedding_fm_6h.slurm

set -eo pipefail

# ---- EDIT THESE FOR YOUR CLUSTER ----
REPO=/orcd/pool/007/pablomer/tess-generative
TORCHENV_PY=/orcd/home/002/pablomer/miniforge3/envs/torchenv/bin/python
DATA_H5=/orcd/pool/007/pablomer/outputs/embedding_fm/anchor_embeddings_100k.h5
RUN_DIR=/orcd/pool/007/pablomer/outputs/embedding_fm/runs/emb-fm-adagn-concat128   # MUST be fixed across resubmits
INJECTION=adagn          # adagn | hybrid | xattn | concat
COND_MODE=concat128      # concat128 | galaxy64 | instrument64
MAX_STEPS=100000
BATCH=128
MAX_RESUBMITS=30
# Provide your W&B key via the environment (do NOT commit it):  export WANDB_API_KEY=...
export WANDB_API_KEY="${WANDB_API_KEY:-}"
# -------------------------------------

mkdir -p "$RUN_DIR/checkpoints"

if [ -f "$RUN_DIR/DONE" ]; then
    echo "[resubmit] $RUN_DIR/DONE present — training complete, exiting."
    exit 0
fi

COUNT_FILE="$RUN_DIR/.resubmit_count"
count=$(cat "$COUNT_FILE" 2>/dev/null || echo 0)
if [ "$count" -ge "$MAX_RESUBMITS" ]; then
    echo "[resubmit] hit MAX_RESUBMITS=$MAX_RESUBMITS — stopping chain."
    exit 1
fi
echo $((count + 1)) > "$COUNT_FILE"

# Queue the successor now (runs after this job ends for ANY reason, incl. timeout kill).
sbatch --dependency=afterany:"$SLURM_JOB_ID" "$0" || echo "[resubmit] could not queue successor"

source ~/.bashrc
conda activate torchenv || true
export PYTHONPATH="$REPO:$PYTHONPATH"
cd "$REPO"

RESUME_ARG=""
if [ -f "$RUN_DIR/checkpoints/last.ckpt" ]; then
    RESUME_ARG="--resume-from $RUN_DIR/checkpoints/last.ckpt"
    echo "[resume] resuming from $RUN_DIR/checkpoints/last.ckpt"
fi

echo "PYTHON=$TORCHENV_PY | INJECTION=$INJECTION | COND_MODE=$COND_MODE"
"$TORCHENV_PY" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

srun "$TORCHENV_PY" -u galaxy_images/galaxy_model/contrastive_baseline/embedding_fm_decoder/train_embedding_fm.py \
    --data-h5 "$DATA_H5" \
    --injection "$INJECTION" \
    --cond-mode "$COND_MODE" \
    --run-dir "$RUN_DIR" \
    --max-steps "$MAX_STEPS" \
    --batch-size "$BATCH" \
    --devices 1 \
    $RESUME_ARG

echo "Training segment finished at $(date)"
