#!/bin/bash
#SBATCH --job-name=snr-traversal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:45:00
#SBATCH --partition=mi2104x
#SBATCH --output=/work1/jeroenaudenaert/pablomer/logs/snr_traversal_%j.out
#SBATCH --error=/work1/jeroenaudenaert/pablomer/logs/snr_traversal_%j.err
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

# User-picked targets from p50–p70 SNR gallery
TARGETS=(92198 53601 91614 29493)

for IDX in "${TARGETS[@]}"; do
    echo "=== Target stats_idx=$IDX ==="
    python3 "$LATENT_DIR/discord_notify.py" "🎯 SNR traversal: starting target stats_idx=$IDX"
    srun python -u galaxy_images/galaxy_model/latent_traversal/snr_conditioning_traversal.py \
        --target-idx "$IDX" \
        --output-dir "$OUTDIR"
done

echo "Job completed at $(date)"
