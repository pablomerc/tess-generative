#!/bin/bash
# One-liner smoke test for the instrument -> physics MLP baseline.
# Allocates a short interactive GPU slot via srun and runs the script inside it.
#
# Usage:  bash smoke_instrument_to_physics.sh

set -eo pipefail

SCRIPT_DIR="/home/pablomer/orcd/pool/tess-generative/galaxy_images/galaxy_model/downstream_evaluation/engaging"
OUT_CSV="${SCRIPT_DIR}/outputs/predictions/predict_instrument_baseline_smoke.csv"

mkdir -p "${SCRIPT_DIR}/outputs/predictions"

srun \
    --job-name=instr2phys-smoke \
    --partition=mit_normal_gpu \
    --gres=gpu:1 \
    --cpus-per-task=4 \
    --mem=16G \
    --time=00:15:00 \
    bash -lc "
        source ~/.bashrc
        conda activate torchenv
        export TORCH_BLAS_PREFER_HIPBLASLT=0
        export OMP_NUM_THREADS=1
        cd '${SCRIPT_DIR}'
        python predict_instrument_to_physics.py \
            --labels-source outputs/embeddings/base.h5 \
            --out '${OUT_CSV}' \
            --device cuda --max-epochs 10 --seed 0
        echo '=== Result ==='
        cat '${OUT_CSV}'
    "
