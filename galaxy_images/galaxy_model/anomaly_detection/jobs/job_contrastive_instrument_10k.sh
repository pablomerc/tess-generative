#!/bin/bash
#SBATCH -J anomaly-cnt-inst-10k
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
export MIOPEN_USER_DB_PATH="/tmp/miopen_${SLURM_JOB_ID}"

DISCORD="https://discord.com/api/webhooks/1496321484338106519/HdI24VGIwsk9IEYoz9MdwMUSmwJ76hJhgIp-TviwYt8Pbnme59KE1xsrHJTM9x3M5eOM"
SUFFIX="contrastive_instrument_10k"
LATENTS="anomaly_detection/outputs/anomaly_latents_ours_contrastive_10k.h5"
OUTDIR="anomaly_detection/outputs"
OUTPNG="anomaly_detection/outputs/figures_compare/compare_top_anomalies_instrument.png"

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

notify "🚀 [job $SLURM_JOB_ID] anomaly-cnt-inst-10k starting..."

# ── Step 1: Fit & Score instrument_emb ──────────────────────────────────────
notify "🧮 [job $SLURM_JOB_ID] Step 1/2: fitting detectors on contrastive instrument_emb (10k)..."
python anomaly_detection/fit_and_score.py \
    --ours "$LATENTS" \
    --suffix "$SUFFIX" \
    --nsf-epochs 50 \
    --also-instrument-emb
if [ $? -ne 0 ]; then notify "❌ [job $SLURM_JOB_ID] fit_and_score FAILED"; exit 1; fi
notify "✅ [job $SLURM_JOB_ID] Step 1/2 done: scores saved to ${OUTDIR}/anomaly_scores_${SUFFIX}.h5"

# ── Step 2: Comparison plot ──────────────────────────────────────────────────
notify "🖼️ [job $SLURM_JOB_ID] Step 2/2: generating instrument comparison plot..."
python anomaly_detection/compare_top_anomalies.py \
    --preset instrument \
    --top-n 12 \
    --out "$OUTPNG"
if [ $? -ne 0 ]; then notify "❌ [job $SLURM_JOB_ID] compare plot FAILED"; exit 1; fi

notify_file "📊 [job $SLURM_JOB_ID] Top-12 instrument anomaly comparison: ours-inst-100k | AION-100k | contrastive-inst-10k" "$OUTPNG"
notify "🎉 [job $SLURM_JOB_ID] anomaly-cnt-inst-10k ALL DONE!"
