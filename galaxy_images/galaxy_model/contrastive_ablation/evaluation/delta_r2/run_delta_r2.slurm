#!/bin/bash
#SBATCH --job-name=delta-r2
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=/orcd/pool/007/pablomer/tess-generative/galaxy_images/galaxy_model/contrastive_ablation/logs/%x-%j.out
#SBATCH --error=/orcd/pool/007/pablomer/tess-generative/galaxy_images/galaxy_model/contrastive_ablation/logs/%x-%j.err

set -euo pipefail

: "${EMBEDDINGS:?export EMBEDDINGS=/path/to/prepare_combined.h5}"
: "${OVERLAP:?export OVERLAP=/path/to/overlap.parquet}"
: "${MODEL_NAME:?export MODEL_NAME=descriptive-model-name}"
: "${OUT_DIR:?export OUT_DIR=/path/to/output-directory}"

# Under sbatch, Slurm copies this file into /var/spool/slurmd/job<id>/, so BASH_SOURCE
# does NOT point at the repo. Resolve the repo location explicitly; SCRIPT_DIR may be
# overridden for out-of-tree checkouts.
SCRIPT_DIR="${SCRIPT_DIR:-/orcd/pool/007/pablomer/tess-generative/galaxy_images/galaxy_model/contrastive_ablation/evaluation/delta_r2}"
[[ -f "${SCRIPT_DIR}/run_delta_r2.py" ]] || { echo "run_delta_r2.py not found under SCRIPT_DIR=${SCRIPT_DIR}" >&2; exit 2; }
PYTHON="${PYTHON:-/orcd/home/002/pablomer/miniforge3/envs/torchenv/bin/python}"

"${PYTHON}" "${SCRIPT_DIR}/run_delta_r2.py" \
  --embeddings "${EMBEDDINGS}" \
  --overlap-parquet "${OVERLAP}" \
  --model-name "${MODEL_NAME}" \
  --out-dir "${OUT_DIR}" \
  "$@"
