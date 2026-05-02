#!/bin/bash
#SBATCH --job-name=engaging-predict
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=4:00:00
##SBATCH --partition=<engaging-gpu-partition>
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --open-mode=append

# Trains MLP probes for every checkpoint H5 produced by run_prepare.slurm,
# then makes the comparison plots.

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

source ~/.bashrc
conda activate torchenv
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export TORCH_BLAS_PREFER_HIPBLASLT=0
export OMP_NUM_THREADS=1

cd "${REPO_ROOT}"

mkdir -p "${SCRIPT_DIR}/outputs/predictions" "${SCRIPT_DIR}/outputs/plots"

shopt -s nullglob
for h5 in "${SCRIPT_DIR}"/outputs/embeddings/*.h5; do
    name="$(basename "${h5}" .h5)"
    echo "=== Predicting for ${name} ==="
    python "${SCRIPT_DIR}/predict_combined.py" \
        --embeddings "${h5}" \
        --out        "${SCRIPT_DIR}/outputs/predictions/predict_${name}.csv"
done

echo "=== Plotting ==="
python "${SCRIPT_DIR}/makeplot.py" \
    --predictions-dir "${SCRIPT_DIR}/outputs/predictions" \
    --out-dir         "${SCRIPT_DIR}/outputs/plots"

echo "Done at $(date)"
