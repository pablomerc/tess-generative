#!/bin/bash
#SBATCH --job-name=qual-trav-batch
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=03:00:00
#SBATCH --partition=mi2104x
#SBATCH --output=/work1/jeroenaudenaert/pablomer/logs/quality_traversal_batch_%j.out
#SBATCH --error=/work1/jeroenaudenaert/pablomer/logs/quality_traversal_batch_%j.err
#SBATCH --open-mode=append

set -eo pipefail

mkdir -p /work1/jeroenaudenaert/pablomer/logs

DISCORD_WEBHOOK="https://discord.com/api/webhooks/1497979386144493680/VA-xWhfTWzc-oeC5EvPzyqEk_MW52wZsK2RyLS0egfhHHHhBxrmb9NGawy0rIpfvn3Zo"

discord_msg() {
    curl -s -X POST "$DISCORD_WEBHOOK" \
        -H "Content-Type: application/json" \
        -d "{\"content\": \"$1\"}" > /dev/null
}

trap 'discord_msg "**Quality traversal batch FAILED** | job=$SLURM_JOB_ID | $(date)"' ERR

discord_msg "**Quality traversal batch $SLURM_JOB_ID started** | 20 random targets | host=$(hostname) | $(date)"

source ~/.bashrc
conda activate torchenv

cd /work1/jeroenaudenaert/pablomer/tess-generative
export PYTHONPATH=/work1/jeroenaudenaert/pablomer/tess-generative:$PYTHONPATH

CACHE=galaxy_images/galaxy_model/latent_traversal/ivar_psf_cache.npz
OUTDIR=galaxy_images/galaxy_model/latent_traversal/quality_traversal_results

# 20 random target indices drawn with seed 42 from 0..101724
# Generated with: python -c "import numpy as np; np.random.seed(42); print(*np.random.randint(0,101725,20))"
TARGETS=(15795 860 76820 54886 6265 82386 37194 87498 44131 60263 16023 41090 67221 64820 769 59735 62955 64925 67969 5311)

TOTAL=${#TARGETS[@]}
COUNT=0

for IDX in "${TARGETS[@]}"; do
    COUNT=$((COUNT + 1))
    echo "=== Target $COUNT/$TOTAL: stats_idx=${IDX} ==="
    discord_msg "**Batch $SLURM_JOB_ID** | target $COUNT/$TOTAL (idx=${IDX})"
    srun python -u galaxy_images/galaxy_model/latent_traversal/quality_conditioning_traversal.py \
        --target-idx ${IDX} \
        --cache-path "${CACHE}" \
        --output-dir "${OUTDIR}" \
        --gallery-discord
done

discord_msg "**Quality traversal batch $SLURM_JOB_ID COMPLETE** | 20 targets done | $(date)"
echo "Job completed at $(date)"
