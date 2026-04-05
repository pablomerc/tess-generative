#!/bin/bash
#SBATCH --job-name=pwr-eval-v2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --partition=mi2104x
#SBATCH --output=/work1/jeroenaudenaert/pablomer/logs/power_eval_v2_%j.out
#SBATCH --error=/work1/jeroenaudenaert/pablomer/logs/power_eval_v2_%j.err
#SBATCH --open-mode=append

set -eo pipefail

mkdir -p /work1/jeroenaudenaert/pablomer/logs

# --- Environment ---
source ~/.bashrc
conda activate torchenv

# --- ROCm workaround: disable hipBLASLt (buggy on MI210 for certain matrix shapes) ---
export TORCH_BLAS_PREFER_HIPBLASLT=0

# --- Run ---
cd /work1/jeroenaudenaert/pablomer/tess-generative

export PYTHONPATH=/work1/jeroenaudenaert/pablomer/tess-generative:$PYTHONPATH

FIGURES_DIR="/work1/jeroenaudenaert/pablomer/tess-generative/galaxy_images/galaxy_model/generative_evaluation/figures_v2"

srun python -u galaxy_images/galaxy_model/generative_evaluation/power_autocorrelation_analysis_v2.py

echo "Job completed at $(date)"

# --- Discord notification ---
DISCORD_WEBHOOK="https://discord.com/api/webhooks/1488692651334177071/8b8KvACfQIVYCNY3ovee04BixCEWiqbqp1iQk4z9sXHlgR29kMkGIjl1pahV5uEPSbxe"
curl -s -F "content=**Job \`${SLURM_JOB_ID}\` done**: power-autocorrelation-v2" \
    "${DISCORD_WEBHOOK}" || true

for FIG in "${FIGURES_DIR}"/individual_sample_*.png; do
    [ -f "${FIG}" ] && curl -s -F "file=@${FIG}" "${DISCORD_WEBHOOK}" || true
done
