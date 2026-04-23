#!/bin/bash
#SBATCH -J pca-tsne-crossspace-joint
#SBATCH -o /work1/jeroenaudenaert/pablomer/logs/job.%j.out
#SBATCH -e /work1/jeroenaudenaert/pablomer/logs/job.%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH -t 06:00:00
#SBATCH -p mi2101x

source ~/.bashrc
conda activate torchenv
export TORCH_BLAS_PREFER_HIPBLASLT=0
export OMP_NUM_THREADS=16

DISCORD="https://discord.com/api/webhooks/1496321484338106519/HdI24VGIwsk9IEYoz9MdwMUSmwJ76hJhgIp-TviwYt8Pbnme59KE1xsrHJTM9x3M5eOM"
OUTDIR="anomaly_detection/outputs"

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
notify "🚀 [job $SLURM_JOB_ID] pca-tsne-crossspace-joint starting (anomalies included in fit, 10k & 100k)"

run_scale() {
    local N_BG=$1
    local SUFFIX="crossspace_${N_BG}_joint"
    local FIGDIR="${OUTDIR}/figures_${SUFFIX}"
    mkdir -p "$FIGDIR"

    notify "▶️ [job $SLURM_JOB_ID] n_background=${N_BG} (joint) ..."

    python anomaly_detection/umap_visualizations/plot_pca_tsne_cross_space.py \
        --ours-latents "${OUTDIR}/anomaly_latents_ours_ours_100k.h5" \
        --aion-latents "${OUTDIR}/anomaly_latents_aion_aion_100k.h5" \
        --ins-latents  "${OUTDIR}/anomaly_latents_ours_ins_100k.h5" \
        --ours-scores  "${OUTDIR}/anomaly_scores_ours_100k.h5" \
        --aion-scores  "${OUTDIR}/anomaly_scores_aion_100k.h5" \
        --ins-scores   "${OUTDIR}/anomaly_scores_ins_100k.h5" \
        --suffix       "$SUFFIX" \
        --n-background "$N_BG" \
        --top-n 30 \
        --n-jobs 16 \
        --include-anomalies

    if [ $? -ne 0 ]; then
        notify "❌ [job $SLURM_JOB_ID] FAILED at n_background=${N_BG} (joint)"
        exit 1
    fi

    notify_file "🗺️ [${SUFFIX}] PCA — physics & AION (joint)"    "${FIGDIR}/pca_crossspace_physics_aion_${SUFFIX}.png"
    notify_file "🗺️ [${SUFFIX}] PCA — instrument (joint)"         "${FIGDIR}/pca_crossspace_instrument_${SUFFIX}.png"
    notify_file "🗺️ [${SUFFIX}] t-SNE — physics & AION (joint)"  "${FIGDIR}/tsne_crossspace_physics_aion_${SUFFIX}.png"
    notify_file "🗺️ [${SUFFIX}] t-SNE — instrument (joint)"       "${FIGDIR}/tsne_crossspace_instrument_${SUFFIX}.png"

    notify "✅ [job $SLURM_JOB_ID] n_background=${N_BG} (joint) done"
}

run_scale 10000
run_scale 100000

notify "🎉 [job $SLURM_JOB_ID] pca-tsne-crossspace-joint ALL DONE!"
