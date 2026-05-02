#!/bin/bash
#SBATCH -J lens-nn-extended
#SBATCH -o /work1/jeroenaudenaert/pablomer/logs/job.%j.out
#SBATCH -e /work1/jeroenaudenaert/pablomer/logs/job.%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH -t 02:00:00
#SBATCH -p mi2101x

source ~/.bashrc
conda activate torchenv
export TORCH_BLAS_PREFER_HIPBLASLT=0
export MIOPEN_USER_DB_PATH="/tmp/miopen_${SLURM_JOB_ID}"
export PYTHONPATH=/work1/jeroenaudenaert/pablomer/tess-generative:$PYTHONPATH

DISCORD="https://discord.com/api/webhooks/1498090322062737558/kASxqetBZeCav7Bxcl3Xz9qlt5FMqzpCJhi5CzCf8LHz9pOVksvIMmCaBufdyxY8Ans-"

GALAXY_MODEL_DIR="/work1/jeroenaudenaert/pablomer/tess-generative/galaxy_images/galaxy_model"
CHECKPOINT="${GALAXY_MODEL_DIR}/outputs/neighbors_all_attn/2026-04-05/checkpoints/best-epoch=228-step=87000.ckpt"
LATENTS="${GALAXY_MODEL_DIR}/neighbor_search/neighbor_latents_extended.h5"
OUT_DIR="${GALAXY_MODEL_DIR}/neighbor_search_lenses/outputs_extended"

notify() {
    curl -s -X POST "$DISCORD" \
        -H "Content-Type: application/json" \
        -d "{\"content\": \"$1\"}" > /dev/null
}

cd "$GALAXY_MODEL_DIR"

# --- Generate extended latents if not present ---
if [ ! -f "$LATENTS" ]; then
    notify "🔭 [job ${SLURM_JOB_ID}] Building extended latents (all 468k entries, source_type 0+1+2)..."
    python neighbor_search_lenses/make_latents_extended.py \
        --checkpoint "$CHECKPOINT" \
        --output "$LATENTS" \
        --batch-size 256
    if [ $? -ne 0 ]; then
        notify "❌ [job ${SLURM_JOB_ID}] make_latents_extended.py FAILED"
        exit 1
    fi
    notify "✅ [job ${SLURM_JOB_ID}] Extended latents ready: $LATENTS"
else
    echo "Extended latents already exist: $LATENTS"
fi

# --- Run lens neighbor search ---
notify "🔭 [job ${SLURM_JOB_ID}] Starting extended lens neighbor search (21 lenses, 569k gallery)..."

python neighbor_search_lenses/search_lens_neighbors.py \
    --latents "$LATENTS" \
    --checkpoint "$CHECKPOINT" \
    --out-dir "$OUT_DIR"

if [ $? -ne 0 ]; then
    notify "❌ [job ${SLURM_JOB_ID}] search_lens_neighbors.py FAILED"
    exit 1
fi

notify "✅ [job ${SLURM_JOB_ID}] Extended search done — sending figures..."
for f in "${OUT_DIR}"/lens_*.png; do
    fname=$(basename "$f")
    curl -s -X POST "$DISCORD" \
        -F "payload_json={\"content\": \"**${fname}** (extended 569k gallery)\"}" \
        -F "file=@${f}" > /dev/null
done
notify "📬 [job ${SLURM_JOB_ID}] All extended figures sent."
