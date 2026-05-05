#!/bin/bash
#SBATCH -J lens-hsc-leg-aion
#SBATCH -o /work1/jeroenaudenaert/pablomer/logs/job.%j.out
#SBATCH -e /work1/jeroenaudenaert/pablomer/logs/job.%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH -t 04:00:00
#SBATCH -p mi2104x

source ~/.bashrc
conda activate torchenv
export TORCH_BLAS_PREFER_HIPBLASLT=0
export MIOPEN_USER_DB_PATH="/tmp/miopen_${SLURM_JOB_ID}"
export PYTHONPATH=/work1/jeroenaudenaert/pablomer/tess-generative:$PYTHONPATH

DISCORD="https://discord.com/api/webhooks/1498090322062737558/kASxqetBZeCav7Bxcl3Xz9qlt5FMqzpCJhi5CzCf8LHz9pOVksvIMmCaBufdyxY8Ans-"
GALAXY_MODEL_DIR="/work1/jeroenaudenaert/pablomer/tess-generative/galaxy_images/galaxy_model"
AION_HSC_LATENTS="${GALAXY_MODEL_DIR}/anomaly_detection/outputs/anomaly_latents_aion_hsc_extended.h5"
AION_LEG_LATENTS="${GALAXY_MODEL_DIR}/anomaly_detection/outputs/anomaly_latents_aion_legacy_extended.h5"
OUT_DIR="${GALAXY_MODEL_DIR}/neighbor_search_lenses/outputs_combined"

notify() {
    curl -s -X POST "$DISCORD" \
        -H "Content-Type: application/json" \
        -d "{\"content\": \"$1\"}" > /dev/null
}

cd "$GALAXY_MODEL_DIR"

# --- Encode AION HSC latents if not present ---
if [ ! -f "$AION_HSC_LATENTS" ]; then
    notify "🔭 [job ${SLURM_JOB_ID}] Encoding AION HSC latents (source_type 0+1, ~366k)..."
    python anomaly_detection/encode_latents_aion_extended.py --survey hsc
    if [ $? -ne 0 ]; then
        notify "❌ [job ${SLURM_JOB_ID}] AION HSC encoding FAILED"
        exit 1
    fi
    notify "✅ [job ${SLURM_JOB_ID}] AION HSC latents ready"
else
    echo "AION HSC latents already exist: $AION_HSC_LATENTS"
fi

# --- Encode AION Legacy latents if not present ---
if [ ! -f "$AION_LEG_LATENTS" ]; then
    notify "🔭 [job ${SLURM_JOB_ID}] Encoding AION Legacy latents (source_type 0+2, ~205k)..."
    python anomaly_detection/encode_latents_aion_extended.py --survey legacy
    if [ $? -ne 0 ]; then
        notify "❌ [job ${SLURM_JOB_ID}] AION Legacy encoding FAILED"
        exit 1
    fi
    notify "✅ [job ${SLURM_JOB_ID}] AION Legacy latents ready"
else
    echo "AION Legacy latents already exist: $AION_LEG_LATENTS"
fi

# --- Run HSC+Legacy combined search ---
notify "🔭 [job ${SLURM_JOB_ID}] Running HSC+Legacy combined search (Lens 48, ours vs AION, L2 + Cosine)..."
python neighbor_search_lenses/search_lens_hsc_aion.py \
    --mode combined \
    --lens-index 48 \
    --out-dir "$OUT_DIR"

if [ $? -ne 0 ]; then
    notify "❌ [job ${SLURM_JOB_ID}] Combined search FAILED"
    exit 1
fi

notify "✅ [job ${SLURM_JOB_ID}] Combined search done — sending figure..."
FIG="${OUT_DIR}/lens_048_hsc_aion_combined.png"
curl -s -X POST "$DISCORD" \
    -F "payload_json={\"content\": \"**Lens 48 — HSC+Legacy combined gallery (~571k) | ours vs AION | L2 + Cosine**\"}" \
    -F "file=@${FIG}" > /dev/null
notify "📬 [job ${SLURM_JOB_ID}] Combined done."
