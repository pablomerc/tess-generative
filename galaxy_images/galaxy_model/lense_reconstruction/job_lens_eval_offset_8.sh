#!/bin/bash
#SBATCH -J lens-eval-offset-8
#SBATCH -o /work1/jeroenaudenaert/pablomer/logs/job.%j.out
#SBATCH -e /work1/jeroenaudenaert/pablomer/logs/job.%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH -t 00:30:00
#SBATCH -p mi2101x

source ~/.bashrc
conda activate torchenv
export TORCH_BLAS_PREFER_HIPBLASLT=0
export MIOPEN_USER_DB_PATH="/tmp/miopen_${SLURM_JOB_ID}"
export PYTHONPATH=/work1/jeroenaudenaert/pablomer/tess-generative:$PYTHONPATH

DISCORD="${DISCORD:-https://discord.com/api/webhooks/1496321484338106519/HdI24VGIwsk9IEYoz9MdwMUSmwJ76hJhgIp-TviwYt8Pbnme59KE1xsrHJTM9x3M5eOM}"
CKPT="${CKPT:-/work1/jeroenaudenaert/pablomer/tess-generative/galaxy_images/galaxy_model/hierarchical_attention/outputs/neighbors_hier_global_ins/2026-04-22_2/checkpoints/latest-step=step=201000.ckpt}"
OUTDIR="${OUTDIR:-lense_reconstruction/outputs/lens_eval_offset_neighbors_8}"

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

notify "🔭 [job $SLURM_JOB_ID] lens-eval-offset-8 starting — ranks 6-10, 8 lenses (red border = random HSC fallback)..."

python lense_reconstruction/eval_hier_global_ins.py \
    --checkpoint "$CKPT" \
    --num-lenses 8 \
    --neighbor-start 5 \
    --out-dir "$OUTDIR" \
    --steps 100

if [ $? -ne 0 ]; then
    notify "❌ [job $SLURM_JOB_ID] lens-eval-offset-8 FAILED — check logs"
    exit 1
fi

notify_file "🌌 [job $SLURM_JOB_ID] Lens eval offset (8 lenses, ranks 6-10) — normalized" \
    "${OUTDIR}/lens_eval_normalized.png"
notify_file "🌌 [job $SLURM_JOB_ID] Lens eval offset (8 lenses, ranks 6-10) — row-scaled" \
    "${OUTDIR}/lens_eval_row_scaled.png"
notify "✅ [job $SLURM_JOB_ID] lens-eval-offset-8 DONE"
