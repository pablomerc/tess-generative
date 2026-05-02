#!/bin/bash
#SBATCH -J lens-final-fig
#SBATCH -o /work1/jeroenaudenaert/pablomer/logs/job.%j.out
#SBATCH -e /work1/jeroenaudenaert/pablomer/logs/job.%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH -t 00:30:00
#SBATCH -p mi2101x

source ~/.bashrc
conda activate torchenv
export TORCH_BLAS_PREFER_HIPBLASLT=0
export MIOPEN_USER_DB_PATH="/tmp/miopen_${SLURM_JOB_ID}"
export PYTHONPATH=/work1/jeroenaudenaert/pablomer/tess-generative:$PYTHONPATH

DISCORD="https://discord.com/api/webhooks/1498090322062737558/kASxqetBZeCav7Bxcl3Xz9qlt5FMqzpCJhi5CzCf8LHz9pOVksvIMmCaBufdyxY8Ans-"

notify() {
    curl -s -X POST "$DISCORD" \
        -H "Content-Type: application/json" \
        -d "{\"content\": \"$1\"}" > /dev/null
}

GALAXY_MODEL_DIR="/work1/jeroenaudenaert/pablomer/tess-generative/galaxy_images/galaxy_model"
OUT_FIG="${GALAXY_MODEL_DIR}/neighbor_search_lenses/outputs/lens_neighbors_final_figure.png"

cd "$GALAXY_MODEL_DIR"

notify "🔭 [job ${SLURM_JOB_ID}] Generating lens final figure (lenses 33 & 48)..."

python neighbor_search_lenses/lens_final_figure.py

if [ $? -ne 0 ]; then
    notify "❌ [job ${SLURM_JOB_ID}] lens_final_figure.py FAILED"
    exit 1
fi

curl -s -X POST "$DISCORD" \
    -F "payload_json={\"content\": \"**Lens neighbors final figure** — Lens 33 (NNs #1,2,3,6,7,8) | Lens 48 (NNs #1,3,4,5,6,8) [display corrected -1]\"}" \
    -F "file=@${OUT_FIG}" > /dev/null

notify "✅ [job ${SLURM_JOB_ID}] Figure sent."
