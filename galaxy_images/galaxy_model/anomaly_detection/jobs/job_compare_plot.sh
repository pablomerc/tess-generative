#!/bin/bash
#SBATCH -J compare-anomaly-plot
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
OUTPNG="anomaly_detection/outputs/figures_compare/compare_top_anomalies_hsc_mean_flow.png"

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

notify "🖼️ [job $SLURM_JOB_ID] compare-anomaly-plot starting..."

python anomaly_detection/compare_top_anomalies.py --top-n 12 --out "$OUTPNG"
if [ $? -ne 0 ]; then
    notify "❌ [job $SLURM_JOB_ID] compare plot FAILED"
    exit 1
fi

notify_file "📊 [job $SLURM_JOB_ID] Top-8 anomaly comparison (hsc_mean/flow): ours-100k | AION-100k | contrastive-10k" "$OUTPNG"
notify "✅ [job $SLURM_JOB_ID] compare-anomaly-plot DONE"
