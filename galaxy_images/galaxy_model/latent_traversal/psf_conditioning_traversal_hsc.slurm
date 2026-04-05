#!/bin/bash
#SBATCH --job-name=psf-traversal-hsc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=03:00:00
#SBATCH --partition=mi2104x
#SBATCH --output=/work1/jeroenaudenaert/pablomer/logs/psf_traversal_hsc_%j.out
#SBATCH --error=/work1/jeroenaudenaert/pablomer/logs/psf_traversal_hsc_%j.err
#SBATCH --open-mode=append

set -eo pipefail

mkdir -p /work1/jeroenaudenaert/pablomer/logs

source ~/.bashrc
conda activate torchenv

export TORCH_BLAS_PREFER_HIPBLASLT=0

RUN_DIR="/work1/jeroenaudenaert/pablomer/tess-generative/galaxy_images/galaxy_model/latent_traversal/latent_traversal_hsc_v2"
mkdir -p "${RUN_DIR}"

cd /work1/jeroenaudenaert/pablomer/tess-generative
export PYTHONPATH=/work1/jeroenaudenaert/pablomer/tess-generative:$PYTHONPATH

TARGET_INDICES=(0 5000 10000 15000 20000 25000 30000 35000 40000 45000 50000 55000 60000 65000 70000 75000 80000 85000 90000 95000)

for TARGET_IDX in "${TARGET_INDICES[@]}"; do
    echo "--- Running HSC traversal for target_idx=${TARGET_IDX} (bucket-size=5) ---"
    python -u galaxy_images/galaxy_model/latent_traversal/psf_conditioning_traversal_hsc.py \
        --target-idx "${TARGET_IDX}" \
        --n-examples 100000 \
        --bucket-size 5 \
        --output-dir "${RUN_DIR}"

    echo "--- Running HSC traversal for target_idx=${TARGET_IDX} (bucket-size=15) ---"
    python -u galaxy_images/galaxy_model/latent_traversal/psf_conditioning_traversal_hsc.py \
        --target-idx "${TARGET_IDX}" \
        --n-examples 100000 \
        --bucket-size 15 \
        --output-dir "${RUN_DIR}"
done

echo "All 20 HSC targets done at $(date)"
