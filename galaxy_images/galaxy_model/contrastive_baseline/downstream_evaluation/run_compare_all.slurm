#!/bin/bash
#SBATCH --job-name=cnt-compare
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=0:30:00
#SBATCH --partition=mi2104x
#SBATCH --output=/work1/jeroenaudenaert/pablomer/logs/compare_all_%j.out
#SBATCH --error=/work1/jeroenaudenaert/pablomer/logs/compare_all_%j.err
#SBATCH --open-mode=append

set -eo pipefail

WEBHOOK="https://discord.com/api/webhooks/1498060835438858251/o5UIoYuSM0JqIR8PXEI7kGTY9eg9sO0Vc9G89WrWbtteAi0WB1cgDif18r0rj4y-Qwix"
SCRIPT_DIR="/work1/jeroenaudenaert/pablomer/tess-generative/galaxy_images/galaxy_model/contrastive_baseline/downstream_evaluation"

discord_msg() { curl -s -X POST "$WEBHOOK" -H "Content-Type: application/json" -d "{\"content\": \"$1\"}" > /dev/null; }
discord_file() { curl -s -X POST "$WEBHOOK" -F "content=$1" -F "file=@$2" > /dev/null; }

on_exit() {
    EXIT_CODE=$?
    [ $EXIT_CODE -ne 0 ] && discord_msg "❌ **compare-all** CRASHED (exit $EXIT_CODE) — job \`$SLURM_JOB_ID\`\nLog: \`/work1/jeroenaudenaert/pablomer/logs/compare_all_${SLURM_JOB_ID}.err\`"
}
trap on_exit EXIT

mkdir -p /work1/jeroenaudenaert/pablomer/logs
source ~/.bashrc
conda activate torchenv

cd /work1/jeroenaudenaert/pablomer/tess-generative
export PYTHONPATH=/work1/jeroenaudenaert/pablomer/tess-generative:$PYTHONPATH

discord_msg "🎨 **compare-all** generating comparison plots — job \`$SLURM_JOB_ID\`"

python "${SCRIPT_DIR}/compare_variants.py" \
  --csv "${SCRIPT_DIR}/predict_all_contrastive_contrastive_300k.csv"  --label "Neighbors, no phys-neg (300k)" \
  --csv "${SCRIPT_DIR}/predict_all_contrastive_contrastive_varA.csv"  --label "Neighbors + phys-neg (100k)" \
  --csv "${SCRIPT_DIR}/predict_all_contrastive_contrastive_varB.csv"  --label "Random pos, no phys-neg (100k)" \
  --csv "${SCRIPT_DIR}/predict_all_contrastive_contrastive_varC.csv"  --label "Random pos + phys-neg (100k)" \
  --output-dir "${SCRIPT_DIR}" \
  --diagnostic

discord_msg "✅ **compare-all** done — job \`$SLURM_JOB_ID\`\nSending plots..."

for png in "${SCRIPT_DIR}"/compare_variants_*.png "${SCRIPT_DIR}"/diagnostic_*.png; do
    [ -f "$png" ] && discord_file "📊 $(basename $png)" "$png"
done

echo "Job completed at $(date)"
