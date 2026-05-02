#!/bin/bash
#SBATCH --job-name=hsc-prep
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=6:00:00
#SBATCH --partition=mi2104x
#SBATCH --output=/work1/jeroenaudenaert/pablomer/logs/hsc_prepare_%j.out
#SBATCH --error=/work1/jeroenaudenaert/pablomer/logs/hsc_prepare_%j.err
#SBATCH --open-mode=append

set -eo pipefail

# ---- Configure these for each run ----
WEBHOOK="https://discord.com/api/webhooks/1498060835438858251/o5UIoYuSM0JqIR8PXEI7kGTY9eg9sO0Vc9G89WrWbtteAi0WB1cgDif18r0rj4y-Qwix"
CHECKPOINT="/work1/jeroenaudenaert/pablomer/outputs/<path-to-your-checkpoint>.ckpt"
MODULE="double_train_fm_neighbors.py"
SUFFIX="<your-suffix>"
# --------------------------------------

SCRIPT_DIR="/work1/jeroenaudenaert/pablomer/tess-generative/galaxy_images/galaxy_model/downstream_evaluation/new_data_downstream"
JOB_NAME="hsc-prepare-${SUFFIX}"

discord_msg() {
    curl -s -X POST "$WEBHOOK" \
        -H "Content-Type: application/json" \
        -d "{\"content\": \"$1\"}" > /dev/null
}

on_exit() {
    EXIT_CODE=$?
    [ $EXIT_CODE -ne 0 ] && discord_msg \
        "❌ **${JOB_NAME}** CRASHED (exit $EXIT_CODE) — job \`$SLURM_JOB_ID\`
Log: \`/work1/jeroenaudenaert/pablomer/logs/hsc_prepare_${SLURM_JOB_ID}.err\`"
}
trap on_exit EXIT

mkdir -p /work1/jeroenaudenaert/pablomer/logs
source ~/.bashrc
conda activate torchenv
export TORCH_BLAS_PREFER_HIPBLASLT=0
export PYTHONPATH=/work1/jeroenaudenaert/pablomer/tess-generative:$PYTHONPATH

cd /work1/jeroenaudenaert/pablomer/tess-generative

discord_msg "🚀 **${JOB_NAME}** started — job \`$SLURM_JOB_ID\` on \`$SLURM_NODELIST\`"

python "${SCRIPT_DIR}/prepare_hsc_downstream.py" \
    --checkpoint "${CHECKPOINT}" \
    --module     "${MODULE}"     \
    --suffix     "${SUFFIX}"     \
    --output-dir "${SCRIPT_DIR}" \
    --batch-size 256

discord_msg "✅ **${JOB_NAME}** finished — job \`$SLURM_JOB_ID\`"
echo "Job completed at $(date)"
