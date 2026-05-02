#!/bin/bash
#SBATCH --job-name=cnt-varA
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --time=8:00:00
#SBATCH --partition=mi2104x
#SBATCH --output=/work1/jeroenaudenaert/pablomer/logs/contrastive_varA_%j.out
#SBATCH --error=/work1/jeroenaudenaert/pablomer/logs/contrastive_varA_%j.err
#SBATCH --open-mode=append

# Variant A: neighbors positives + physics pair as explicit instrument negative

set -eo pipefail

WEBHOOK="https://discord.com/api/webhooks/1498060835438858251/o5UIoYuSM0JqIR8PXEI7kGTY9eg9sO0Vc9G89WrWbtteAi0WB1cgDif18r0rj4y-Qwix"
JOB_NAME="varA (neighbors + phys-neg)"

discord_msg() {
    curl -s -X POST "$WEBHOOK" \
        -H "Content-Type: application/json" \
        -d "{\"content\": \"$1\"}" > /dev/null
}

on_exit() {
    EXIT_CODE=$?
    if [ $EXIT_CODE -ne 0 ]; then
        discord_msg "❌ **${JOB_NAME}** CRASHED (exit $EXIT_CODE) — job \`$SLURM_JOB_ID\`\nLog: \`/work1/jeroenaudenaert/pablomer/logs/contrastive_varA_${SLURM_JOB_ID}.err\`"
    fi
}
trap on_exit EXIT

mkdir -p /work1/jeroenaudenaert/pablomer/logs
mkdir -p /work1/jeroenaudenaert/pablomer/outputs/contrastive_baseline

source ~/.bashrc
conda activate torchenv

export TORCH_BLAS_PREFER_HIPBLASLT=0
export N_DEVICES=4

discord_msg "🚀 **${JOB_NAME}** started — job \`$SLURM_JOB_ID\` on \`$SLURM_NODELIST\`\n100k steps, neighbors pos + physics pair as instrument negative"

cd /work1/jeroenaudenaert/pablomer/tess-generative
export PYTHONPATH=/work1/jeroenaudenaert/pablomer/tess-generative:$PYTHONPATH

srun python -u galaxy_images/galaxy_model/contrastive_baseline/train_dual_encoder_contrastive_varA.py

discord_msg "✅ **${JOB_NAME}** finished — job \`$SLURM_JOB_ID\`\nCheckpoint in \`/work1/jeroenaudenaert/pablomer/outputs/contrastive_baseline/\`"

echo "Job completed at $(date)"
