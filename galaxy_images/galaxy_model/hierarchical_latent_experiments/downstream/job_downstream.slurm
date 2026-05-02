#!/bin/bash
#SBATCH -J hier-downstream
#SBATCH -o /work1/jeroenaudenaert/pablomer/logs/hier_lat_down_%A_%a.out
#SBATCH -e /work1/jeroenaudenaert/pablomer/logs/hier_lat_down_%A_%a.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH -t 06:00:00
#SBATCH -p mi2101x
#SBATCH --array=0-3

set -eo pipefail

source ~/.bashrc
conda activate torchenv
set -u
ulimit -n 65536 || true
export TORCH_BLAS_PREFER_HIPBLASLT=0
export MIOPEN_USER_DB_PATH="/tmp/miopen_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
export MIOPEN_CUSTOM_CACHE_DIR="$MIOPEN_USER_DB_PATH"
mkdir -p "$MIOPEN_USER_DB_PATH"
export PYTHONPATH=/work1/jeroenaudenaert/pablomer/tess-generative:${PYTHONPATH:-}
export MPLCONFIGDIR=/tmp/matplotlib_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}
export NUMBA_CACHE_DIR=/tmp/numba_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}

DISCORD="https://discord.com/api/webhooks/1499879903464788150/EiosG8MxzpVp4aGsb9FkxV7J-xEpZMABK4yTiwSBdEZ5GA7onfIKvBtHJOaPGli8SGv5"
CKPT="${CKPT:-/work1/jeroenaudenaert/pablomer/tess-generative/galaxy_images/galaxy_model/hierarchical_attention/outputs/neighbors_hier_global_ins/2026-04-22_2/checkpoints/latest-step=step=201000.ckpt}"
ROOT=/work1/jeroenaudenaert/pablomer/tess-generative/galaxy_images/galaxy_model/hierarchical_latent_experiments

TAG="${TAG:-}"
VARIANTS=(spatial_flat global_vec spatial_pooled global_concat)
V=${VARIANTS[$SLURM_ARRAY_TASK_ID]}
OUT_NAME="$V${TAG:+__$TAG}"
SUFFIX="hier_${V}${TAG:+_$TAG}"
OUT="$ROOT/downstream/outputs/$OUT_NAME"
mkdir -p "$OUT"

notify() {
    curl -s -X POST "$DISCORD" -H "Content-Type: application/json" \
        -d "{\"content\": \"$1\"}" > /dev/null || true
}

trap 'rc=$?; if [ $rc -ne 0 ]; then notify "💥 [downstream/$V job $SLURM_JOB_ID.$SLURM_ARRAY_TASK_ID] hard FAIL rc=$rc — check logs"; fi' EXIT

cd /work1/jeroenaudenaert/pablomer/tess-generative

notify "🛰️ [downstream/$V job $SLURM_JOB_ID.$SLURM_ARRAY_TASK_ID] starting (suffix=$SUFFIX)"

python -u -m galaxy_images.galaxy_model.hierarchical_latent_experiments.downstream.prepare \
    --variant "$V" --ckpt "$CKPT" --suffix "$SUFFIX" --output-dir "$OUT"

python -u -m galaxy_images.galaxy_model.hierarchical_latent_experiments.downstream.predict \
    --variant "$V" --suffix "$SUFFIX" --output-dir "$OUT"
