#!/bin/bash
#SBATCH -J base6x6-umap-pairs
#SBATCH -o /work1/jeroenaudenaert/pablomer/logs/base6x6_umap_%A_%a.out
#SBATCH -e /work1/jeroenaudenaert/pablomer/logs/base6x6_umap_%A_%a.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH -t 01:30:00
#SBATCH -p mi2101x
#SBATCH --array=0-1

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
CKPT="/work1/jeroenaudenaert/pablomer/tess-generative/galaxy_images/galaxy_model/checkpoints/base6x6/best-epoch=216-step=83000.ckpt"
ROOT=/work1/jeroenaudenaert/pablomer/tess-generative/galaxy_images/galaxy_model/base6x6_experiments
N="${N:-4096}"

VARIANTS=(spatial_pooled spatial_flat)
V=${VARIANTS[$SLURM_ARRAY_TASK_ID]}

notify() {
    curl -s -X POST "$DISCORD" -H "Content-Type: application/json" \
        -d "{\"content\": \"$1\"}" > /dev/null || true
}

trap 'rc=$?; if [ $rc -ne 0 ]; then notify "💥 [base6x6/umap_pairs/$V job $SLURM_JOB_ID.$SLURM_ARRAY_TASK_ID] hard FAIL rc=$rc — check logs"; fi' EXIT

cd /work1/jeroenaudenaert/pablomer/tess-generative

notify "🛰️ [base6x6/umap_pairs/$V job $SLURM_JOB_ID.$SLURM_ARRAY_TASK_ID] starting (N=$N)"

python -u -m galaxy_images.galaxy_model.base6x6_experiments.umap_pairs.run_umap_pairs \
    --variant "$V" \
    --n "$N" \
    --ckpt "$CKPT" \
    --out-dir "$ROOT/umap_pairs/outputs/base6x6/$V"
