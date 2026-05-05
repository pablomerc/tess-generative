#!/bin/bash
#SBATCH -J aion-hsc-dedup
#SBATCH -o /work1/jeroenaudenaert/pablomer/logs/job.%j.out
#SBATCH -e /work1/jeroenaudenaert/pablomer/logs/job.%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH -t 01:00:00
#SBATCH -p mi2101x

source ~/.bashrc
conda activate torchenv

DISCORD="https://discord.com/api/webhooks/1496321484338106519/HdI24VGIwsk9IEYoz9MdwMUSmwJ76hJhgIp-TviwYt8Pbnme59KE1xsrHJTM9x3M5eOM"
SUFFIX="aion_367k_hsc_only_dedup10"
SCORES="anomaly_detection/outputs/anomaly_scores_aion_367k.h5"
OUT_DIR="anomaly_detection/outputs/figures_${SUFFIX}"
LABEL_PREFIX="aion-367k HSC dedup10\""

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

if [ ! -f "$SCORES" ]; then
    notify "❌ [job $SLURM_JOB_ID] AION HSC-only dedup post: scores file missing ($SCORES). Did 304748 succeed?"
    exit 1
fi

notify "📊 [job $SLURM_JOB_ID] AION 367k: producing HSC-only RA/Dec-deduplicated (>10\") top-anomaly grids..."

python anomaly_detection/top_anomalies_hsc_only.py \
    --scores "$SCORES" \
    --suffix "$SUFFIX" \
    --out-dir "$OUT_DIR" \
    --top-n 25 \
    --dedup-arcsec 10.0 \
    --candidate-multiplier 500
if [ $? -ne 0 ]; then
    notify "❌ [job $SLURM_JOB_ID] HSC-only dedup plotting FAILED"
    exit 1
fi

notify "✅ [job $SLURM_JOB_ID] HSC-only dedup plots ready — uploading."

for PNG in "${OUT_DIR}"/top_anomalies_*.png; do
    LABEL=$(basename "$PNG" .png | sed -e "s/^top_anomalies_//" -e "s/_${SUFFIX}$//")
    notify_file "📊 [${LABEL_PREFIX}] ${LABEL}" "$PNG"
    sleep 1
done

notify "🎉 [job $SLURM_JOB_ID] AION 367k HSC-only dedup post DONE."
