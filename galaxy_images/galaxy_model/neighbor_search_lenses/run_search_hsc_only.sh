#!/bin/bash
#SBATCH -J search-lens-hsc-only
#SBATCH -o /work1/jeroenaudenaert/pablomer/logs/job.%j.out
#SBATCH -e /work1/jeroenaudenaert/pablomer/logs/job.%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH -t 01:00:00
#SBATCH -p mi2104x

source ~/.bashrc
conda activate torchenv
export TORCH_BLAS_PREFER_HIPBLASLT=0
export MIOPEN_USER_DB_PATH="/tmp/miopen_${SLURM_JOB_ID}"
export OPENBLAS_NUM_THREADS=64
export PYTHONPATH=/work1/jeroenaudenaert/pablomer/tess-generative:$PYTHONPATH

DISCORD="https://discord.com/api/webhooks/1498090322062737558/kASxqetBZeCav7Bxcl3Xz9qlt5FMqzpCJhi5CzCf8LHz9pOVksvIMmCaBufdyxY8Ans-"
GALAXY_MODEL_DIR="/work1/jeroenaudenaert/pablomer/tess-generative/galaxy_images/galaxy_model"
LENS_INDEX="${LENS_INDEX:-48}"   # set via --export=ALL,LENS_INDEX=N, default 48
OUT_DIR="${GALAXY_MODEL_DIR}/neighbor_search_lenses/outputs_hsc_only"

notify() {
    curl -s -X POST "$DISCORD" -H "Content-Type: application/json" \
        -d "{\"content\": \"$1\"}" > /dev/null
}

cd "$GALAXY_MODEL_DIR"

notify "🔍 [job ${SLURM_JOB_ID}] Lens ${LENS_INDEX} HSC-only search (ours vs AION, L2 + Cosine)..."

python neighbor_search_lenses/search_lens_hsc_aion.py \
    --mode hsc_only \
    --lens-index "$LENS_INDEX" \
    --out-dir "$OUT_DIR"

if [ $? -ne 0 ]; then
    notify "❌ [job ${SLURM_JOB_ID}] HSC-only search FAILED"
    exit 1
fi

FIG="${OUT_DIR}/lens_$(printf '%03d' $LENS_INDEX)_hsc_aion_hsc_only.png"
curl -s -X POST "$DISCORD" \
    -F "payload_json={\"content\": \"**Lens ${LENS_INDEX} — HSC-only (~366k) | ours vs AION | L2 + Cosine**\"}" \
    -F "file=@${FIG}" > /dev/null
notify "📬 [job ${SLURM_JOB_ID}] HSC-only done."
