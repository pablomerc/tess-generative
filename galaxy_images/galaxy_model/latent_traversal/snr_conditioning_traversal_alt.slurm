#!/bin/bash
#SBATCH --job-name=snr-traversal-alt
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --partition=mi2104x
#SBATCH --output=/work1/jeroenaudenaert/pablomer/logs/snr_traversal_alt_%j.out
#SBATCH --error=/work1/jeroenaudenaert/pablomer/logs/snr_traversal_alt_%j.err
#SBATCH --open-mode=append

set -eo pipefail

mkdir -p /work1/jeroenaudenaert/pablomer/logs

source ~/.bashrc
conda activate torchenv

LATENT_DIR=/work1/jeroenaudenaert/pablomer/tess-generative/galaxy_images/galaxy_model/latent_traversal
source "$LATENT_DIR/_slurm_notify_wrapper.sh"

cd /work1/jeroenaudenaert/pablomer/tess-generative
export PYTHONPATH=/work1/jeroenaudenaert/pablomer/tess-generative:$PYTHONPATH
export TORCH_BLAS_PREFER_HIPBLASLT=0

OUTDIR=/work1/jeroenaudenaert/pablomer/tess-generative/galaxy_images/galaxy_model/latent_traversal/snr_traversal_results

# Same 4 targets as the k5 run
TARGETS=(92198 53601 91614 29493)

# Two alternative neighbor configurations
for MODE in k10 k1x5; do
    for IDX in "${TARGETS[@]}"; do
        echo "=== mode=$MODE  target stats_idx=$IDX ==="
        python3 "$LATENT_DIR/discord_notify.py" "🎯 SNR traversal [$MODE]: starting target stats_idx=$IDX"
        srun python -u galaxy_images/galaxy_model/latent_traversal/snr_conditioning_traversal.py \
            --target-idx "$IDX" \
            --mode "$MODE" \
            --output-dir "$OUTDIR"
    done
done

echo "Job completed at $(date)"
