#!/bin/bash
#SBATCH -J flow-anomaly
#SBATCH -o /work1/jeroenaudenaert/pablomer/logs/job.%j.out
#SBATCH -e /work1/jeroenaudenaert/pablomer/logs/job.%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH -t 04:00:00
#SBATCH -p mi2101x

# Usage: sbatch job_flow.sh <suffix> <ours|aion|both>
# e.g.:  sbatch job_flow.sh ours_10k ours
#        sbatch job_flow.sh aion_100k aion

source ~/.bashrc
conda activate torchenv
export TORCH_BLAS_PREFER_HIPBLASLT=0

DISCORD="https://discord.com/api/webhooks/1496321484338106519/HdI24VGIwsk9IEYoz9MdwMUSmwJ76hJhgIp-TviwYt8Pbnme59KE1xsrHJTM9x3M5eOM"
SUFFIX="${1:-ours_10k}"
MODE="${2:-ours}"
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
notify "🌊 [job $SLURM_JOB_ID] NSF flow training starting — suffix=${SUFFIX}, mode=${MODE}"

mkdir -p "$FIGDIR"

# Build args
ARGS="--suffix $SUFFIX"
if [ "$MODE" = "ours" ] || [ "$MODE" = "both" ]; then
    ARGS="$ARGS --ours ${OUTDIR}/anomaly_latents_ours_${SUFFIX}.h5"
fi
if [ "$MODE" = "aion" ] || [ "$MODE" = "both" ]; then
    ARGS="$ARGS --aion ${OUTDIR}/anomaly_latents_aion_${SUFFIX}.h5"
fi

python anomaly_detection/run_flow_only.py $ARGS
if [ $? -ne 0 ]; then notify "❌ [job $SLURM_JOB_ID] flow training FAILED (${SUFFIX})"; exit 1; fi
notify "✅ [job $SLURM_JOB_ID] flow scores computed — visualizing..."

# Visualize flow images only
DATA="/work1/jeroenaudenaert/pablomer/data/neighbours_v2.h5"
OURS_ARG=""
if [ "$MODE" = "ours" ] || [ "$MODE" = "both" ]; then
    OURS_ARG="--ours ${OUTDIR}/anomaly_latents_ours_${SUFFIX}.h5"
fi

python anomaly_detection/visualize_top_anomalies.py \
    --scores "${OUTDIR}/anomaly_scores_${SUFFIX}.h5" \
    $OURS_ARG \
    --data "$DATA" \
    --suffix "$SUFFIX" \
    --top-n 25
if [ $? -ne 0 ]; then notify "❌ [job $SLURM_JOB_ID] visualize FAILED (${SUFFIX})"; exit 1; fi

# Send only flow images to Discord
for PNG in "${FIGDIR}"/top_anomalies_*_flow_*.png "${FIGDIR}"/top_anomalies_*flow*.png; do
    [ -f "$PNG" ] || continue
    LABEL=$(basename "$PNG" .png | sed "s/top_anomalies_//" | sed "s/_${SUFFIX}//")
    notify_file "🌊 [${SUFFIX}] ${LABEL}" "$PNG"
done

notify "🎉 [job $SLURM_JOB_ID] flow done for ${SUFFIX}!"
