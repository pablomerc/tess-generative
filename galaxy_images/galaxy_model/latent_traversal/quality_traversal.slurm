#!/bin/bash
#SBATCH --job-name=qual-traversal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --partition=mi2104x
#SBATCH --output=/work1/jeroenaudenaert/pablomer/logs/quality_traversal_%j.out
#SBATCH --error=/work1/jeroenaudenaert/pablomer/logs/quality_traversal_%j.err
#SBATCH --open-mode=append

set -eo pipefail

mkdir -p /work1/jeroenaudenaert/pablomer/logs

DISCORD_WEBHOOK="https://discord.com/api/webhooks/1497979386144493680/VA-xWhfTWzc-oeC5EvPzyqEk_MW52wZsK2RyLS0egfhHHHhBxrmb9NGawy0rIpfvn3Zo"

discord_msg() {
    curl -s -X POST "$DISCORD_WEBHOOK" \
        -H "Content-Type: application/json" \
        -d "{\"content\": \"$1\"}" > /dev/null
}

# Notify on failure
trap 'discord_msg "**Quality traversal FAILED** | job=$SLURM_JOB_ID | $(date)"' ERR

discord_msg "**Quality traversal job $SLURM_JOB_ID queued/started** | host=$(hostname) | $(date)"

source ~/.bashrc
conda activate torchenv

cd /work1/jeroenaudenaert/pablomer/tess-generative
export PYTHONPATH=/work1/jeroenaudenaert/pablomer/tess-generative:$PYTHONPATH

CACHE=galaxy_images/galaxy_model/latent_traversal/ivar_psf_cache.npz
OUTDIR=galaxy_images/galaxy_model/latent_traversal/quality_traversal_results

# 5 target galaxies spread across the quality range (p10, p30, p50, p70, p90)
for PCT in 10 30 50 70 90; do
    echo "=== Target at quality p${PCT} ==="
    discord_msg "**Starting target p${PCT}** | job=$SLURM_JOB_ID"
    srun python -u galaxy_images/galaxy_model/latent_traversal/quality_conditioning_traversal.py \
        --target-quality-pct ${PCT} \
        --cache-path "${CACHE}" \
        --output-dir "${OUTDIR}"
done

discord_msg "**Quality traversal job $SLURM_JOB_ID COMPLETE** | all 5 targets done | $(date)"
echo "Job completed at $(date)"
