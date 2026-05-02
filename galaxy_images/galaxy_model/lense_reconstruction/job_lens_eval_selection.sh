#!/bin/bash
#SBATCH -J lens-eval-selection
#SBATCH -o /work1/jeroenaudenaert/pablomer/logs/job.%j.out
#SBATCH -e /work1/jeroenaudenaert/pablomer/logs/job.%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH -t 01:00:00
#SBATCH -p mi2101x

set -eo pipefail
source ~/.bashrc
conda activate torchenv
ulimit -n 65536 || true
export TORCH_BLAS_PREFER_HIPBLASLT=0
export MIOPEN_USER_DB_PATH="/tmp/miopen_${SLURM_JOB_ID}"
export PYTHONPATH=/work1/jeroenaudenaert/pablomer/tess-generative:$PYTHONPATH

DISCORD="${DISCORD:-https://discord.com/api/webhooks/1499879903464788150/EiosG8MxzpVp4aGsb9FkxV7J-xEpZMABK4yTiwSBdEZ5GA7onfIKvBtHJOaPGli8SGv5}"
CKPT="${CKPT:-/work1/jeroenaudenaert/pablomer/tess-generative/galaxy_images/galaxy_model/checkpoints/hier-small/snapshot.ckpt}"
OUTDIR="${OUTDIR:-lense_reconstruction/outputs/selection_figures_hier_small_ranks_1-5}"
NEIGHBOR_START="${NEIGHBOR_START:-0}"
LENS_INDICES="${LENS_INDICES:-5,8,12,18,20,29,32,33,40,41,44,46,48,49,53,63,64,66,67,68,70}"
LABEL="${LABEL:-ranks_1-5}"

notify() {
    curl -s -X POST "$DISCORD" \
        -H "Content-Type: application/json" \
        -d "{\"content\": \"$1\"}" > /dev/null
}

notify_file() {
    if [ -f "$2" ]; then
        curl -s -X POST "$DISCORD" \
            -F "payload_json={\"content\": \"$1\"}" \
            -F "file=@$2" > /dev/null
    else
        notify "$1 (file not found: $2)"
    fi
}

cd /work1/jeroenaudenaert/pablomer/tess-generative/galaxy_images/galaxy_model

notify "🔭 [job $SLURM_JOB_ID] selection-figures ($LABEL) hier-small starting — 21 lenses, neighbor_start=$NEIGHBOR_START"

python lense_reconstruction/eval_hier_global_ins.py \
    --checkpoint "$CKPT" \
    --lens-indices "$LENS_INDICES" \
    --neighbor-start "$NEIGHBOR_START" \
    --out-dir "$OUTDIR" \
    --steps 100

if [ $? -ne 0 ]; then
    notify "❌ [job $SLURM_JOB_ID] selection-figures ($LABEL) FAILED — check logs"
    exit 1
fi

notify_file "🌌 [job $SLURM_JOB_ID] selection_$LABEL hier-small — normalized" "${OUTDIR}/lens_eval_normalized.png"
notify_file "🌌 [job $SLURM_JOB_ID] selection_$LABEL hier-small — row-scaled" "${OUTDIR}/lens_eval_row_scaled.png"
notify "✅ [job $SLURM_JOB_ID] selection-figures ($LABEL) DONE"
