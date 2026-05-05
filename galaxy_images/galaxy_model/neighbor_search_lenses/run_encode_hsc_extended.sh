#!/bin/bash
#SBATCH -J encode-aion-hsc-ext
#SBATCH -o /work1/jeroenaudenaert/pablomer/logs/job.%j.out
#SBATCH -e /work1/jeroenaudenaert/pablomer/logs/job.%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH -t 04:00:00
#SBATCH -p mi2104x

source ~/.bashrc
conda activate torchenv
export TORCH_BLAS_PREFER_HIPBLASLT=0
export MIOPEN_USER_DB_PATH="/tmp/miopen_${SLURM_JOB_ID}"
export PYTHONPATH=/work1/jeroenaudenaert/pablomer/tess-generative:$PYTHONPATH

DISCORD="https://discord.com/api/webhooks/1498090322062737558/kASxqetBZeCav7Bxcl3Xz9qlt5FMqzpCJhi5CzCf8LHz9pOVksvIMmCaBufdyxY8Ans-"
GALAXY_MODEL_DIR="/work1/jeroenaudenaert/pablomer/tess-generative/galaxy_images/galaxy_model"

notify() {
    curl -s -X POST "$DISCORD" -H "Content-Type: application/json" \
        -d "{\"content\": \"$1\"}" > /dev/null
}

cd "$GALAXY_MODEL_DIR"

notify "🔭 [job ${SLURM_JOB_ID}] Encoding AION HSC extended (source_type 0+1, ~366k)..."

python anomaly_detection/encode_latents_aion_extended.py \
    --survey hsc \
    --source-types 0 1 \
    --suffix hsc_extended

if [ $? -ne 0 ]; then
    notify "❌ [job ${SLURM_JOB_ID}] AION HSC extended encoding FAILED"
    exit 1
fi

notify "✅ [job ${SLURM_JOB_ID}] AION HSC extended done (anomaly_latents_aion_hsc_extended.h5)"
