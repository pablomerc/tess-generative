#!/bin/bash
#SBATCH -J paper-anomaly-367k
#SBATCH -o /work1/jeroenaudenaert/pablomer/logs/job.%j.out
#SBATCH -e /work1/jeroenaudenaert/pablomer/logs/job.%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH -t 00:30:00
#SBATCH -p mi2101x

source ~/.bashrc
conda activate torchenv

DISCORD="https://discord.com/api/webhooks/1496321484338106519/HdI24VGIwsk9IEYoz9MdwMUSmwJ76hJhgIp-TviwYt8Pbnme59KE1xsrHJTM9x3M5eOM"
SUFFIX="367k"
OUT_DIR="anomaly_detection/outputs/figures_compare_${SUFFIX}"

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

notify "🖼️ [job $SLURM_JOB_ID] paper-anomaly-${SUFFIX} starting (3-column figure: Physics | AION | Instrument over 366,706 examples)..."

# Sanity-check that all three score files exist before plotting.
for KIND in ours aion ins; do
    F="anomaly_detection/outputs/anomaly_scores_${KIND}_${SUFFIX}.h5"
    if [ ! -f "$F" ]; then
        notify "❌ [job $SLURM_JOB_ID] missing input: $F. Run job_${KIND}_${SUFFIX}.sh first."
        exit 1
    fi
done

python anomaly_detection/paper_anomaly_figure.py --suffix "$SUFFIX" --out-dir "$OUT_DIR"
if [ $? -ne 0 ]; then
    notify "❌ [job $SLURM_JOB_ID] paper_anomaly_figure FAILED"
    exit 1
fi

for N in 8 9 12; do
    PNG="${OUT_DIR}/paper_anomaly_${N}.png"
    notify_file "📊 [${SUFFIX}] paper_anomaly_${N}" "$PNG"
done

notify "✅ [job $SLURM_JOB_ID] paper-anomaly-${SUFFIX} DONE"
