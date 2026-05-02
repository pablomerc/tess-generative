#!/bin/bash
#SBATCH -J hier-anom
#SBATCH -o /work1/jeroenaudenaert/pablomer/logs/hier_lat_anom_%A_%a.out
#SBATCH -e /work1/jeroenaudenaert/pablomer/logs/hier_lat_anom_%A_%a.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH -t 04:00:00
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

DISCORD="https://discord.com/api/webhooks/1499879903464788150/EiosG8MxzpVp4aGsb9FkxV7J-xEpZMABK4yTiwSBdEZ5GA7onfIKvBtHJOaPGli8SGv5"
CKPT="${CKPT:-/work1/jeroenaudenaert/pablomer/tess-generative/galaxy_images/galaxy_model/hierarchical_attention/outputs/neighbors_hier_global_ins/2026-04-22_2/checkpoints/latest-step=step=201000.ckpt}"
ROOT=/work1/jeroenaudenaert/pablomer/tess-generative/galaxy_images/galaxy_model/hierarchical_latent_experiments
N_MAX="${N_MAX:-8000}"
NSF_EPOCHS="${NSF_EPOCHS:-50}"
TAG="${TAG:-}"

VARIANTS=(spatial_flat global_vec spatial_pooled global_concat)
V=${VARIANTS[$SLURM_ARRAY_TASK_ID]}
OUT_NAME="$V${TAG:+__$TAG}"
OUT="$ROOT/anomalies/outputs/$OUT_NAME"
mkdir -p "$OUT"

notify() {
    curl -s -X POST "$DISCORD" -H "Content-Type: application/json" \
        -d "{\"content\": \"$1\"}" > /dev/null || true
}

trap 'rc=$?; if [ $rc -ne 0 ]; then notify "💥 [anomalies/$V job $SLURM_JOB_ID.$SLURM_ARRAY_TASK_ID] hard FAIL rc=$rc — check logs"; fi' EXIT

cd /work1/jeroenaudenaert/pablomer/tess-generative

notify "🛰️ [anomalies/$V job $SLURM_JOB_ID.$SLURM_ARRAY_TASK_ID] starting (N_MAX=$N_MAX, NSF_EPOCHS=$NSF_EPOCHS)"

python -u -m galaxy_images.galaxy_model.hierarchical_latent_experiments.anomalies.encode_latents \
    --variant "$V" --ckpt "$CKPT" --out-path "$OUT/latents.h5" --n-max "$N_MAX"

python -u -m galaxy_images.galaxy_model.hierarchical_latent_experiments.anomalies.fit_and_score \
    --latents "$OUT/latents.h5" --out-path "$OUT/scores.h5" --variant "$V" \
    --nsf-epochs "$NSF_EPOCHS"

python -u -m galaxy_images.galaxy_model.hierarchical_latent_experiments.anomalies.visualize \
    --scores "$OUT/scores.h5" --latents "$OUT/latents.h5" \
    --variant "$V" --out-dir "$OUT/figures"
