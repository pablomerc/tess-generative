#!/bin/bash
#SBATCH -J anomaly-aion-10k-fit
#SBATCH -o /work1/jeroenaudenaert/pablomer/logs/job.%j.out
#SBATCH -e /work1/jeroenaudenaert/pablomer/logs/job.%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH -t 04:00:00
#SBATCH -p mi2101x

source ~/.bashrc
conda activate torchenv
export TORCH_BLAS_PREFER_HIPBLASLT=0

DISCORD="https://discord.com/api/webhooks/1496321484338106519/HdI24VGIwsk9IEYoz9MdwMUSmwJ76hJhgIp-TviwYt8Pbnme59KE1xsrHJTM9x3M5eOM"
SUFFIX="aion_10k"
OUTDIR="anomaly_detection/outputs"
FIGDIR="anomaly_detection/outputs/figures_${SUFFIX}"

notify() {
    curl -s -X POST "$DISCORD" \
        -H "Content-Type: application/json" \
        -d "{\"content\": \"$1\"}" > /dev/null
}

notify_file() {
    local MSG="$1"
    local FILE="$2"
    if [ -f "$FILE" ]; then
        curl -s -X POST "$DISCORD" \
            -F "content=$MSG" \
            -F "file=@$FILE" > /dev/null
    else
        notify "$MSG (file not found: $FILE)"
    fi
}

cd /work1/jeroenaudenaert/pablomer/tess-generative/galaxy_images/galaxy_model
notify "🔧 [job $SLURM_JOB_ID] anomaly-aion-10k fit+visualize only (using existing latents)"

mkdir -p "$FIGDIR"

# ── Step 2: Fit & Score ──────────────────────────────────────────────────────
notify "🧮 [job $SLURM_JOB_ID] Step 1/2: fitting detectors on AION latents (PCA→64D)..."
python anomaly_detection/fit_and_score.py \
    --aion "${OUTDIR}/anomaly_latents_aion_${SUFFIX}.h5" \
    --suffix "$SUFFIX" \
    --nsf-epochs 50
if [ $? -ne 0 ]; then notify "❌ [job $SLURM_JOB_ID] fit_and_score FAILED"; exit 1; fi
notify "✅ [job $SLURM_JOB_ID] Step 1/2 done: scores saved"

# ── Step 3: Visualize ────────────────────────────────────────────────────────
notify "🖼️ [job $SLURM_JOB_ID] Step 2/2: generating anomaly images..."
python anomaly_detection/visualize_top_anomalies.py \
    --scores "${OUTDIR}/anomaly_scores_${SUFFIX}.h5" \
    --data   /work1/jeroenaudenaert/pablomer/data/neighbours_v2.h5 \
    --suffix "$SUFFIX" \
    --top-n 25
if [ $? -ne 0 ]; then notify "❌ [job $SLURM_JOB_ID] visualize FAILED"; exit 1; fi

notify "✅ [job $SLURM_JOB_ID] Step 2/2 done. Sending plots..."
for PNG in "${FIGDIR}"/top_anomalies_*.png; do
    LABEL=$(basename "$PNG" .png | sed "s/top_anomalies_//" | sed "s/_${SUFFIX}//")
    notify_file "📊 [${SUFFIX}] ${LABEL}" "$PNG"
done
notify_file "📈 [${SUFFIX}] Spearman rank correlation" "${FIGDIR}/score_correlation_${SUFFIX}.png"
UMAP="${FIGDIR}/umap_anomaly_scores_${SUFFIX}.png"
if [ -f "$UMAP" ]; then notify_file "🗺️ [${SUFFIX}] UMAP" "$UMAP"; fi

notify "🎉 [job $SLURM_JOB_ID] anomaly-aion-10k fit+visualize ALL DONE!"
