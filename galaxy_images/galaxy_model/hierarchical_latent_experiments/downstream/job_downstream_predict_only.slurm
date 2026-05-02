#!/bin/bash
#SBATCH -J hier-down-predict
#SBATCH -o /work1/jeroenaudenaert/pablomer/logs/hier_lat_down_pred_%A_%a.out
#SBATCH -e /work1/jeroenaudenaert/pablomer/logs/hier_lat_down_pred_%A_%a.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH -t 01:00:00
#SBATCH -p mi2101x
#SBATCH --array=0-3

set -eo pipefail

source ~/.bashrc
conda activate torchenv
set -u
export TORCH_BLAS_PREFER_HIPBLASLT=0
export MIOPEN_USER_DB_PATH="/tmp/miopen_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
export MIOPEN_CUSTOM_CACHE_DIR="$MIOPEN_USER_DB_PATH"
mkdir -p "$MIOPEN_USER_DB_PATH"
export PYTHONPATH=/work1/jeroenaudenaert/pablomer/tess-generative:${PYTHONPATH:-}
export MPLCONFIGDIR=/tmp/matplotlib_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}
export NUMBA_CACHE_DIR=/tmp/numba_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}

DISCORD="https://discord.com/api/webhooks/1499879903464788150/EiosG8MxzpVp4aGsb9FkxV7J-xEpZMABK4yTiwSBdEZ5GA7onfIKvBtHJOaPGli8SGv5"
ROOT=/work1/jeroenaudenaert/pablomer/tess-generative/galaxy_images/galaxy_model/hierarchical_latent_experiments

VARIANTS=(spatial_flat global_vec spatial_pooled global_concat)
V=${VARIANTS[$SLURM_ARRAY_TASK_ID]}
SUFFIX="hier_${V}"
OUT="$ROOT/downstream/outputs/$V"

notify() {
    curl -s -X POST "$DISCORD" -H "Content-Type: application/json" \
        -d "{\"content\": \"$1\"}" > /dev/null || true
}

trap 'rc=$?; if [ $rc -ne 0 ]; then notify "💥 [downstream-predict-only/$V job $SLURM_JOB_ID.$SLURM_ARRAY_TASK_ID] hard FAIL rc=$rc — check logs"; fi' EXIT

cd /work1/jeroenaudenaert/pablomer/tess-generative

notify "🛰️ [downstream-predict-only/$V job $SLURM_JOB_ID.$SLURM_ARRAY_TASK_ID] starting (suffix=$SUFFIX)"

python -u -m galaxy_images.galaxy_model.hierarchical_latent_experiments.downstream.predict \
    --variant "$V" --suffix "$SUFFIX" --output-dir "$OUT"
