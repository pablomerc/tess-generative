#!/bin/bash
#SBATCH -J anomaly-cnt-10k
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
SUFFIX="contrastive_10k"
OUTDIR="anomaly_detection/outputs"
FIGDIR="anomaly_detection/outputs/figures_${SUFFIX}"
CKPT_SEARCH_DIR="/work1/jeroenaudenaert/pablomer/outputs/contrastive_baseline"

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

# ── Find best checkpoint ─────────────────────────────────────────────────────
CKPT=$(find "$CKPT_SEARCH_DIR" -name "best-epoch=*.ckpt" | sort | tail -1)
if [ -z "$CKPT" ]; then
    notify "❌ [job $SLURM_JOB_ID] anomaly-cnt-10k: no best checkpoint found in ${CKPT_SEARCH_DIR}"
    exit 1
fi
notify "🚀 [job $SLURM_JOB_ID] anomaly-cnt-10k starting with checkpoint: $(basename $CKPT)"

# ── Step 1: Encode ──────────────────────────────────────────────────────────
notify "🔢 [job $SLURM_JOB_ID] Step 1/3: encoding 10k examples with contrastive model..."
python anomaly_detection/encode_latents_contrastive.py \
    --checkpoint "$CKPT" \
    --suffix "$SUFFIX" \
    --batch-size 256 \
    --n-max 10000
if [ $? -ne 0 ]; then notify "❌ [job $SLURM_JOB_ID] encode FAILED"; exit 1; fi
notify "✅ [job $SLURM_JOB_ID] Step 1/3 done: latents saved to ${OUTDIR}/anomaly_latents_ours_${SUFFIX}.h5"

# ── Step 2: Fit & Score ──────────────────────────────────────────────────────
notify "🧮 [job $SLURM_JOB_ID] Step 2/3: fitting detectors (kNN, IF, Mahal, NSF)..."
python anomaly_detection/fit_and_score.py \
    --ours "${OUTDIR}/anomaly_latents_ours_${SUFFIX}.h5" \
    --suffix "$SUFFIX" \
    --nsf-epochs 50
if [ $? -ne 0 ]; then notify "❌ [job $SLURM_JOB_ID] fit_and_score FAILED"; exit 1; fi
notify "✅ [job $SLURM_JOB_ID] Step 2/3 done: scores saved to ${OUTDIR}/anomaly_scores_${SUFFIX}.h5"

# ── Step 3: Visualize ────────────────────────────────────────────────────────
notify "🖼️ [job $SLURM_JOB_ID] Step 3/3: generating anomaly images..."
python anomaly_detection/visualize_top_anomalies.py \
    --scores "${OUTDIR}/anomaly_scores_${SUFFIX}.h5" \
    --ours   "${OUTDIR}/anomaly_latents_ours_${SUFFIX}.h5" \
    --data   /work1/jeroenaudenaert/pablomer/data/neighbours_v2.h5 \
    --suffix "$SUFFIX" \
    --top-n 25
if [ $? -ne 0 ]; then notify "❌ [job $SLURM_JOB_ID] visualize FAILED"; exit 1; fi

notify "✅ [job $SLURM_JOB_ID] Step 3/3 done: figures in ${FIGDIR}. Sending plots..."

for PNG in "${FIGDIR}"/top_anomalies_*.png; do
    LABEL=$(basename "$PNG" .png)
    notify_file "📊 [${SUFFIX}] ${LABEL}" "$PNG"
done
notify_file "📈 [${SUFFIX}] Spearman rank correlation" "${FIGDIR}/score_correlation_${SUFFIX}.png"
UMAP="${FIGDIR}/umap_anomaly_scores_${SUFFIX}.png"
if [ -f "$UMAP" ]; then notify_file "🗺️ [${SUFFIX}] UMAP colored by scores" "$UMAP"; fi

notify "🎉 [job $SLURM_JOB_ID] anomaly-cnt-10k ALL DONE!"
