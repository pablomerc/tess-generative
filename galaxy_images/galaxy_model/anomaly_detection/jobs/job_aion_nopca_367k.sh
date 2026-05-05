#!/bin/bash
#SBATCH -J aion-nopca-367k
#SBATCH -o /work1/jeroenaudenaert/pablomer/logs/job.%j.out
#SBATCH -e /work1/jeroenaudenaert/pablomer/logs/job.%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH -t 12:00:00
#SBATCH -p mi2104x

source ~/.bashrc
conda activate torchenv
export TORCH_BLAS_PREFER_HIPBLASLT=0
export MIOPEN_USER_DB_PATH="/tmp/miopen_${SLURM_JOB_ID}"
export MIOPEN_DISABLE_CACHE=0
mkdir -p "$MIOPEN_USER_DB_PATH"

DISCORD="https://discord.com/api/webhooks/1496321484338106519/HdI24VGIwsk9IEYoz9MdwMUSmwJ76hJhgIp-TviwYt8Pbnme59KE1xsrHJTM9x3M5eOM"
SUFFIX="aion_nopca_367k"
LATENTS="anomaly_detection/outputs/anomaly_latents_aion_aion_367k.h5"
SCORES="anomaly_detection/outputs/anomaly_scores_${SUFFIX}.h5"
PLOT_SUFFIX="aion_nopca_367k_hsc_only_dedup10"
OUT_DIR="anomaly_detection/outputs/figures_${PLOT_SUFFIX}"

notify() {
    curl -s -X POST "$DISCORD" -H "Content-Type: application/json" \
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

if [ ! -f "$LATENTS" ]; then
    notify "❌ [job $SLURM_JOB_ID] aion-nopca-367k: latents file missing ($LATENTS)"
    exit 1
fi

notify "🚀 [job $SLURM_JOB_ID] aion-nopca-367k starting: NSF + IForest on raw 768-d AION (no PCA), then HSC-only dedup10 plots"

# ── Step 1: Fit NSF + IForest on raw 768-d AION ─────────────────────────────
notify "🧮 [job $SLURM_JOB_ID] Step 1/2: fitting NSF + IsolationForest on full 768-d AION embeddings..."
python anomaly_detection/fit_aion_no_pca.py \
    --aion "$LATENTS" \
    --suffix "$SUFFIX" \
    --nsf-epochs 50
if [ $? -ne 0 ]; then
    notify "❌ [job $SLURM_JOB_ID] fit_aion_no_pca FAILED"
    exit 1
fi
notify "✅ [job $SLURM_JOB_ID] Step 1/2 done: scores saved to ${SCORES}"

# ── Step 2: HSC-only dedup10" grids + Discord post ──────────────────────────
notify "📊 [job $SLURM_JOB_ID] Step 2/2: generating HSC-only RA/Dec-deduplicated (>10\") grids..."
python anomaly_detection/top_anomalies_hsc_only.py \
    --scores "$SCORES" \
    --suffix "$PLOT_SUFFIX" \
    --out-dir "$OUT_DIR" \
    --top-n 25 \
    --dedup-arcsec 10.0 \
    --candidate-multiplier 500
if [ $? -ne 0 ]; then
    notify "❌ [job $SLURM_JOB_ID] HSC-only dedup plotting FAILED"
    exit 1
fi

notify "✅ [job $SLURM_JOB_ID] Step 2/2 done — uploading."
for PNG in "${OUT_DIR}"/top_anomalies_*.png; do
    LABEL=$(basename "$PNG" .png | sed -e "s/^top_anomalies_//" -e "s/_${PLOT_SUFFIX}$//")
    notify_file "📊 [aion-nopca-367k HSC dedup10\"] ${LABEL}" "$PNG"
    sleep 1
done

notify "🎉 [job $SLURM_JOB_ID] aion-nopca-367k DONE."
