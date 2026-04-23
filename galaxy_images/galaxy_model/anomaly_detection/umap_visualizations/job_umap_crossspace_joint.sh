#!/bin/bash
#SBATCH -J umap-crossspace-joint
#SBATCH -o /work1/jeroenaudenaert/pablomer/logs/job.%j.out
#SBATCH -e /work1/jeroenaudenaert/pablomer/logs/job.%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH -t 02:00:00
#SBATCH -p mi2101x

source ~/.bashrc
conda activate torchenv
export TORCH_BLAS_PREFER_HIPBLASLT=0

DISCORD="https://discord.com/api/webhooks/1496321484338106519/HdI24VGIwsk9IEYoz9MdwMUSmwJ76hJhgIp-TviwYt8Pbnme59KE1xsrHJTM9x3M5eOM"
SUFFIX="crossspace_100k_joint"
OUTDIR="anomaly_detection/outputs"
FIGDIR="${OUTDIR}/figures_${SUFFIX}"

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
notify "🚀 [job $SLURM_JOB_ID] umap-crossspace-joint starting (anomalies included in fit)"
mkdir -p "$FIGDIR"

python anomaly_detection/umap_visualizations/plot_umap_cross_space.py \
    --ours-latents "${OUTDIR}/anomaly_latents_ours_ours_100k.h5" \
    --aion-latents "${OUTDIR}/anomaly_latents_aion_aion_100k.h5" \
    --ins-latents  "${OUTDIR}/anomaly_latents_ours_ins_100k.h5" \
    --ours-scores  "${OUTDIR}/anomaly_scores_ours_100k.h5" \
    --aion-scores  "${OUTDIR}/anomaly_scores_aion_100k.h5" \
    --ins-scores   "${OUTDIR}/anomaly_scores_ins_100k.h5" \
    --suffix "$SUFFIX" \
    --n-background 100000 \
    --top-n 30 \
    --include-anomalies

if [ $? -ne 0 ]; then
    notify "❌ [job $SLURM_JOB_ID] umap-crossspace-joint FAILED"
    exit 1
fi

notify_file "🗺️ [${SUFFIX}] Physics & AION anomalies (joint)" \
    "${FIGDIR}/umap_crossspace_physics_aion_${SUFFIX}.png"
notify_file "🗺️ [${SUFFIX}] Instrument anomalies (joint)" \
    "${FIGDIR}/umap_crossspace_instrument_${SUFFIX}.png"

notify "🎉 [job $SLURM_JOB_ID] umap-crossspace-joint ALL DONE!"
