#!/bin/bash
#SBATCH --job-name=cnt-ds-varA
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=4:00:00
#SBATCH --partition=mi2104x
#SBATCH --output=/work1/jeroenaudenaert/pablomer/logs/downstream_varA_%j.out
#SBATCH --error=/work1/jeroenaudenaert/pablomer/logs/downstream_varA_%j.err
#SBATCH --open-mode=append

set -eo pipefail

WEBHOOK="https://discord.com/api/webhooks/1498060835438858251/o5UIoYuSM0JqIR8PXEI7kGTY9eg9sO0Vc9G89WrWbtteAi0WB1cgDif18r0rj4y-Qwix"
JOB_NAME="downstream-varA"
CHECKPOINT="/work1/jeroenaudenaert/pablomer/outputs/contrastive_baseline/dual-encoder-contrastive-varA-physics-neg_20260426_154102/latest-step=step=100000.ckpt"
SUFFIX="contrastive_varA"
NEIGHBORS_HDF5="/work1/jeroenaudenaert/pablomer/data/neighbours_v2.h5"
SCRIPT_DIR="/work1/jeroenaudenaert/pablomer/tess-generative/galaxy_images/galaxy_model/contrastive_baseline/downstream_evaluation"

discord_msg() { curl -s -X POST "$WEBHOOK" -H "Content-Type: application/json" -d "{\"content\": \"$1\"}" > /dev/null; }

on_exit() {
    EXIT_CODE=$?
    [ $EXIT_CODE -ne 0 ] && discord_msg "❌ **${JOB_NAME}** CRASHED (exit $EXIT_CODE) — job \`$SLURM_JOB_ID\`\nLog: \`/work1/jeroenaudenaert/pablomer/logs/downstream_varA_${SLURM_JOB_ID}.err\`"
}
trap on_exit EXIT

mkdir -p /work1/jeroenaudenaert/pablomer/logs
source ~/.bashrc
conda activate torchenv
export TORCH_BLAS_PREFER_HIPBLASLT=0

cd /work1/jeroenaudenaert/pablomer/tess-generative
export PYTHONPATH=/work1/jeroenaudenaert/pablomer/tess-generative:$PYTHONPATH

discord_msg "🚀 **${JOB_NAME}** started — job \`$SLURM_JOB_ID\` on \`$SLURM_NODELIST\`"

python "${SCRIPT_DIR}/prepare_all_contrastive.py" \
  --checkpoint "${CHECKPOINT}" --suffix "${SUFFIX}" \
  --output-dir "${SCRIPT_DIR}" --batch-size 256 --seed 42 \
  --datasets neighbors --neighbors-hdf5 "${NEIGHBORS_HDF5}"

python "${SCRIPT_DIR}/predict_all_contrastive.py" \
  --suffix "${SUFFIX}" --output-dir "${SCRIPT_DIR}" \
  --seed 42 --datasets neighbors

discord_msg "✅ **${JOB_NAME}** finished — job \`$SLURM_JOB_ID\`"
echo "Job completed at $(date)"
