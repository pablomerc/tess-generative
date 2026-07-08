#!/bin/bash
#SBATCH --job-name=engaging-build
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=1:00:00
#SBATCH --partition=mit_normal
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --open-mode=append

set -eo pipefail

SCRIPT_DIR="/home/pablomer/orcd/pool/tess-generative/galaxy_images/galaxy_model/downstream_evaluation/engaging"
REPO_ROOT="/home/pablomer/orcd/pool/tess-generative"

source ~/.bashrc
# NOTE(engaging): replace torchenv with whatever conda env you set up there.
conda activate torchenv
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

cd "${REPO_ROOT}"

python "${SCRIPT_DIR}/build_overlap_dataset.py" \
    --hsc-dir    /home/pablomer/orcd/scratch/hsc_downstream \
    --legacy-dir /home/pablomer/orcd/scratch/legacy_downstream_full/full_1M \
    --out-dir    "${SCRIPT_DIR}/outputs/index" \
    --max-per-subset 16000 \
    --match-radius-arcsec 1.0 \
    --seed 42

echo "Done at $(date)"
