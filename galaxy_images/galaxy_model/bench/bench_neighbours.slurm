#!/bin/bash
#SBATCH --job-name=ne-bench
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:45:00
#SBATCH --partition=mi2104x
#SBATCH --output=/work1/jeroenaudenaert/pablomer/logs/bench_%j.out
#SBATCH --error=/work1/jeroenaudenaert/pablomer/logs/bench_%j.err
#SBATCH --open-mode=append

set -eo pipefail

mkdir -p /work1/jeroenaudenaert/pablomer/logs

# --- Environment ---
source ~/.bashrc
conda activate torchenv

# --- ROCm workaround ---
export TORCH_BLAS_PREFER_HIPBLASLT=0

# --- Run ---
cd /work1/jeroenaudenaert/pablomer/tess-generative

export PYTHONPATH=/work1/jeroenaudenaert/pablomer/tess-generative:$PYTHONPATH

srun python -u galaxy_images/galaxy_model/bench/bench_neighbours.py \
    --h5             /work1/jeroenaudenaert/pablomer/data/neighbor_batches/neighbors_shard_0000.h5 \
    --batch-size     64  \
    --num-workers     0  \
    --warmup-batches  5  \
    --bench-batches  20  \
    --train-steps    10  \
    --output-dir     galaxy_images/galaxy_model/bench/results

echo "Benchmark completed at $(date)"
