#!/bin/bash
#SBATCH -J snr-pct-hist
#SBATCH -o /work1/jeroenaudenaert/pablomer/logs/job.%j.out
#SBATCH -e /work1/jeroenaudenaert/pablomer/logs/job.%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH -t 00:30:00
#SBATCH -p mi2101x

source ~/.bashrc
source /work1/jeroenaudenaert/pablomer/miniconda3/etc/profile.d/conda.sh
conda activate torchenv
PYTHON=/work1/jeroenaudenaert/pablomer/miniconda3/envs/torchenv/bin/python

DISCORD="https://discord.com/api/webhooks/1496321484338106519/HdI24VGIwsk9IEYoz9MdwMUSmwJ76hJhgIp-TviwYt8Pbnme59KE1xsrHJTM9x3M5eOM"
TOP_N=512

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

notify "📈 [job $SLURM_JOB_ID] SNR-percentile histograms (top-${TOP_N}) starting (flow + iforest)..."

for METHOD in flow iforest; do
    OUT="anomaly_detection/outputs/figures_compare/snr_pct_hist_top${TOP_N}_${METHOD}.png"
    "$PYTHON" anomaly_detection/top_anomalies_snr_histogram.py --method "$METHOD" --top-n "$TOP_N" --out "$OUT"
    if [ $? -ne 0 ]; then
        notify "❌ [job $SLURM_JOB_ID] ${METHOD} histogram FAILED"
        exit 1
    fi
    notify_file "📊 [job $SLURM_JOB_ID] Top-${TOP_N} anomalies SNR-pct histogram (${METHOD}): Physics | AION | Instrument" "$OUT"
done

notify "✅ [job $SLURM_JOB_ID] SNR-percentile histograms DONE"
