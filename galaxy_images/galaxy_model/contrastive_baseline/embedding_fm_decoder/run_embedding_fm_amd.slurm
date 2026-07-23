#!/bin/bash
#SBATCH --job-name=emb-fm
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --partition=mi3001x
#SBATCH --output=/work1/jeroenaudenaert/pablomer/logs/emb_fm/%x_%j.out
#SBATCH --error=/work1/jeroenaudenaert/pablomer/logs/emb_fm/%x_%j.err
#SBATCH --open-mode=append
#
# AMD (ROCm) launcher for the embedding-conditioned FM decoder.
# Runs Stage 1 (precompute, skipped if the H5 already exists) then Stage 2 (train) in one job.
# Parameterize per run with:  sbatch --export=ALL,VAR=... run_embedding_fm_amd.slurm
#   OUT_H5 RUN_DIR LIMIT MAX_STEPS VAL_INTERVAL WANDB_NAME INJECTION COND_MODE BATCH PBATCH
#   CKPT_EVERY  SELF_RESUBMIT(0/1)  MAX_RESUBMITS
# Defaults below = the FULL run.  mi3501x is drained -> we use mi3001x (MI300X).  NOTE: this
# cluster's QOS caps each job at 4h wall, so the script SELF-RESUBMITS (afterany) until DONE.

set -eo pipefail

REPO=/work1/jeroenaudenaert/pablomer/tess-generative
TORCHENV_PY=/work1/jeroenaudenaert/pablomer/miniconda3/envs/torchenv/bin/python
DATA_DIR=/work1/jeroenaudenaert/pablomer/data/neighbors_efficient
CKPT=/work1/jeroenaudenaert/pablomer/outputs/contrastive_baseline/dual-encoder-contrastive-resnet18_20260423_175337/best-epoch=599-step=240000.ckpt

# ---- per-run params (override via --export); defaults = FULL run ----
: "${OUT_H5:=/work1/jeroenaudenaert/pablomer/outputs/embedding_fm/anchor_embeddings_100k.h5}"
: "${RUN_DIR:=/work1/jeroenaudenaert/pablomer/outputs/embedding_fm/runs/emb-fm-adagn-concat128}"
: "${LIMIT:=100000}"
: "${MAX_STEPS:=100000}"
: "${VAL_INTERVAL:=1000}"
: "${WANDB_NAME:=emb-fm-adagn-concat128}"
: "${INJECTION:=adagn}"
: "${COND_MODE:=concat128}"
: "${BATCH:=128}"
: "${PBATCH:=256}"
: "${CKPT_EVERY:=250}"    # periodic last.ckpt cadence (steps) -> caps work lost on a timeout

# AMD MI-series ROCm workarounds (needed on MI210 AND MI300X here):
#  1) disable buggy hipBLASLt (also pair with preferred_blas_library("hipblas") in the .py)
#  2) give MIOpen a writable per-job cache dir (default path -> miopenStatusInternalError on BN/conv)
export TORCH_BLAS_PREFER_HIPBLASLT=0
export MIOPEN_USER_DB_PATH="/tmp/miopen_${USER}_${SLURM_JOB_ID}"
export MIOPEN_CUSTOM_CACHE_DIR="$MIOPEN_USER_DB_PATH"
mkdir -p "$MIOPEN_USER_DB_PATH"

source ~/.bashrc
conda activate torchenv || true
export PYTHONPATH="$REPO:$PYTHONPATH"
export MPLCONFIGDIR="/tmp/mpl_${USER}"
export NUMBA_CACHE_DIR="/tmp/numba_${USER}"
# WANDB auth comes from ~/.netrc; honor WANDB_API_KEY if the caller exports one.
export WANDB_API_KEY="${WANDB_API_KEY:-}"
cd "$REPO"

# Chained segments (4h wall cap): if training already hit max-steps, stop the chain here.
if [ -f "$RUN_DIR/DONE" ]; then
    echo "[chain] $RUN_DIR/DONE present — training complete, exiting."
    exit 0
fi

echo "==== $(date) | host=$(hostname) | partition=$SLURM_JOB_PARTITION | job=$SLURM_JOB_ID ===="
"$TORCHENV_PY" -c "import torch;print('torch',torch.__version__,'| hip',torch.version.hip,'| gpu',torch.cuda.is_available(), (torch.cuda.get_device_name(0) if torch.cuda.is_available() else ''))"

PKG=galaxy_images/galaxy_model/contrastive_baseline/embedding_fm_decoder

# ---- Self-perpetuating resume chain -------------------------------------------------------
# The 4h wall cap means one job rarely reaches max-steps. Queue a successor (afterany) BEFORE
# training, so ANY end-of-job (including a timeout SIGKILL) is followed by a resume. The DONE
# guard above stops the chain once training completes; a counter caps runaway chains. This makes
# reaching max-steps independent of how many 4h segments it takes. Disable with SELF_RESUBMIT=0
# (e.g. short smoke / resume tests that manage their own chaining).
: "${MAX_RESUBMITS:=40}"
if [ "${SELF_RESUBMIT:-1}" = "1" ] && [ -n "${SLURM_JOB_ID:-}" ]; then
    mkdir -p "$RUN_DIR"
    CF="$RUN_DIR/.resubmit_count"
    c=$(cat "$CF" 2>/dev/null || echo 0)
    if [ "$c" -ge "$MAX_RESUBMITS" ]; then
        echo "[chain] hit MAX_RESUBMITS=$MAX_RESUBMITS — NOT queuing a successor (investigate if no DONE)."
    else
        echo $((c + 1)) > "$CF"
        if sbatch --dependency=afterany:"$SLURM_JOB_ID" -J "${SLURM_JOB_NAME:-emb-fm}" -t 04:00:00 \
                  --export=ALL "$REPO/$PKG/run_embedding_fm_amd.slurm" >/dev/null 2>&1; then
            echo "[chain] queued resume successor (afterany:$SLURM_JOB_ID), attempt $((c + 1))/$MAX_RESUBMITS"
        else
            echo "[chain] WARNING: could not queue resume successor"
        fi
    fi
fi

# ---- Stage 1: precompute embeddings (atomic publish + completeness-validated skip) --------
# Guard on the finalize marker (num_examples == LIMIT), not mere existence, and publish via
# .tmp + mv, so a precompute killed mid-write can never be silently reused as a zero-tail dataset.
_h5_complete() {
    "$TORCHENV_PY" - "$1" "$LIMIT" <<'PYEOF' 2>/dev/null
import h5py, sys
p, limit = sys.argv[1], int(sys.argv[2])
try:
    with h5py.File(p, "r") as f:
        n = int(f.attrs.get("num_examples", -1))
        ok = (n == limit) and ("targets" in f) and (f["targets"].shape[0] == n)
    sys.exit(0 if ok else 1)
except Exception:
    sys.exit(1)
PYEOF
}
if [ -f "$OUT_H5" ] && _h5_complete "$OUT_H5"; then
    echo "[stage1] $OUT_H5 present and complete (num_examples=$LIMIT) — skipping precompute"
else
    echo "[stage1] precompute LIMIT=$LIMIT -> $OUT_H5 (atomic via .tmp)"
    rm -f "$OUT_H5.tmp"
    srun "$TORCHENV_PY" -u "$PKG/precompute_embeddings.py" \
        --data-dir "$DATA_DIR" --ckpt "$CKPT" --out-h5 "$OUT_H5.tmp" \
        --limit "$LIMIT" --batch-size "$PBATCH" --num-workers 8
    if _h5_complete "$OUT_H5.tmp"; then
        mv -f "$OUT_H5.tmp" "$OUT_H5"
        echo "[stage1] published complete H5 -> $OUT_H5"
    else
        echo "[stage1] ERROR: precompute produced an incomplete H5 (.tmp kept for inspection) — exiting nonzero"
        exit 1
    fi
fi

# ---- Stage 2: train FM decoder (resume if a last.ckpt is present) ----
mkdir -p "$RUN_DIR/checkpoints"
RESUME_ARG=""
if [ -f "$RUN_DIR/checkpoints/last.ckpt" ]; then
    RESUME_ARG="--resume-from $RUN_DIR/checkpoints/last.ckpt"
    echo "[stage2] resuming from $RUN_DIR/checkpoints/last.ckpt"
fi
echo "[stage2] train INJECTION=$INJECTION COND_MODE=$COND_MODE MAX_STEPS=$MAX_STEPS WANDB_NAME=$WANDB_NAME"
srun "$TORCHENV_PY" -u "$PKG/train_embedding_fm.py" \
    --data-h5 "$OUT_H5" \
    --injection "$INJECTION" --cond-mode "$COND_MODE" \
    --run-dir "$RUN_DIR" --wandb-name "$WANDB_NAME" \
    --max-steps "$MAX_STEPS" --val-check-interval "$VAL_INTERVAL" \
    --ckpt-every-n-steps "$CKPT_EVERY" \
    --batch-size "$BATCH" --devices 1 \
    $RESUME_ARG

echo "==== done $(date) ===="
