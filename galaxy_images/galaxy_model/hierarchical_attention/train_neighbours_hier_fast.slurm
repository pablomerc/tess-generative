#!/bin/bash
#SBATCH --job-name=ne-hier-fast
#SBATCH --nodes=1
#SBATCH --cpus-per-task=64
#SBATCH --time=02:00:00
#SBATCH --partition=mi3508x
#SBATCH --output=/work1/jeroenaudenaert/pablomer/logs/hier_neighbors_%j.out
#SBATCH --error=/work1/jeroenaudenaert/pablomer/logs/hier_neighbors_%j.err
#SBATCH --open-mode=append

set -eo pipefail

mkdir -p /work1/jeroenaudenaert/pablomer/logs

# --- Environment ---
source ~/.bashrc
conda activate torchenv

# --- ROCm workaround (kept for safety, may not be needed on MI350X) ---
export TORCH_BLAS_PREFER_HIPBLASLT=0

# --- WandB ---
export WANDB_API_KEY="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"

# --- Run ---
cd /work1/jeroenaudenaert/pablomer/tess-generative

export PYTHONPATH=/work1/jeroenaudenaert/pablomer/tess-generative:$PYTHONPATH

srun python -u galaxy_images/galaxy_model/hierarchical_attention/neighbours_train_hier.py \
    --devices 4 \
    --batch_size 64 \
    --num_workers 8 \
    --wandb_name "hier-bn_36x16-mi350x" \
    --precision "bf16-mixed"

echo "Job completed at $(date)"
