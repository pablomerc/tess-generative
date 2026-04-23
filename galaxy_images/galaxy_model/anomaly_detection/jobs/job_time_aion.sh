#!/bin/bash
#SBATCH -J time-aion
#SBATCH -o /work1/jeroenaudenaert/pablomer/logs/job.%j.out
#SBATCH -e /work1/jeroenaudenaert/pablomer/logs/job.%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH -t 00:30:00
#SBATCH -p devel

source ~/.bashrc
conda activate torchenv
export TORCH_BLAS_PREFER_HIPBLASLT=0

DISCORD="https://discord.com/api/webhooks/1496321484338106519/HdI24VGIwsk9IEYoz9MdwMUSmwJ76hJhgIp-TviwYt8Pbnme59KE1xsrHJTM9x3M5eOM"

notify() {
    curl -s -X POST "$DISCORD" \
        -H "Content-Type: application/json" \
        -d "{\"content\": \"$1\"}" > /dev/null
}

notify "⏱️ [job $SLURM_JOB_ID] AION timing test starting (5 examples)..."

cd /work1/jeroenaudenaert/pablomer/tess-generative/galaxy_images/galaxy_model

python anomaly_detection/time_aion_encoding.py --n 5 --batch-size 32

EXIT=$?
LOG_OUT="/work1/jeroenaudenaert/pablomer/logs/job.${SLURM_JOB_ID}.out"

if [ $EXIT -eq 0 ]; then
    # Extract the key timing lines from log and send to Discord
    TIMING=$(grep -E "examples in|Per example|Extrapolated" "$LOG_OUT" 2>/dev/null | head -6 | tr '\n' ' | ')
    notify "✅ [job $SLURM_JOB_ID] AION timing done! $TIMING"
else
    notify "❌ [job $SLURM_JOB_ID] AION timing FAILED (exit $EXIT). Check $LOG_OUT"
fi
