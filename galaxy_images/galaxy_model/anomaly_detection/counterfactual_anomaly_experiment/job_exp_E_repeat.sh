#!/bin/bash
#SBATCH -J expE-repeat
#SBATCH -o /work1/jeroenaudenaert/pablomer/logs/job.%j.out
#SBATCH -e /work1/jeroenaudenaert/pablomer/logs/job.%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH -t 01:00:00
#SBATCH -p mi2101x

source ~/.bashrc
conda activate torchenv
export TORCH_BLAS_PREFER_HIPBLASLT=0

DISCORD="https://discord.com/api/webhooks/1496321484338106519/HdI24VGIwsk9IEYoz9MdwMUSmwJ76hJhgIp-TviwYt8Pbnme59KE1xsrHJTM9x3M5eOM"

notify_file() {
    local MSG="$1"
    local FILE="$2"
    if [ -f "$FILE" ]; then
        curl -s -X POST "$DISCORD" \
            -F "content=$MSG" \
            -F "file=@$FILE" > /dev/null
    else
        curl -s -X POST "$DISCORD" \
            -H "Content-Type: application/json" \
            -d "{\"content\": \"$MSG (file not found: $FILE)\"}" > /dev/null
    fi
}

cd /work1/jeroenaudenaert/pablomer/tess-generative/galaxy_images/galaxy_model

python anomaly_detection/counterfactual_anomaly_experiment/run_exp_E_repeat.py

if [ $? -ne 0 ]; then
    curl -s -X POST "$DISCORD" \
        -H "Content-Type: application/json" \
        -d "{\"content\": \"❌ [job $SLURM_JOB_ID] exp_E repeat FAILED\"}" > /dev/null
    exit 1
fi

notify_file "🎲 [counterfactual] E (repeat) - 10 instr x 10 random HSC, single + mean(5)" \
    "anomaly_detection/counterfactual_anomaly_experiment/outputs/exp_E_repeat_random_10pairs/artifact_correction_random_10pairs.png"
