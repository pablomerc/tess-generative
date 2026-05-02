#!/bin/bash
#SBATCH --job-name=engaging-prepare
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=8:00:00
##SBATCH --partition=<engaging-gpu-partition>
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --open-mode=append

# Sweeps prepare_combined.py over all four registered checkpoints.
# Each variant has its own (module, model_class) — see ../../checkpoints/README.md.

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
CKPT_ROOT="${REPO_ROOT}/galaxy_images/galaxy_model/checkpoints"

source ~/.bashrc
conda activate torchenv
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export TORCH_BLAS_PREFER_HIPBLASLT=0
export OMP_NUM_THREADS=1

cd "${REPO_ROOT}"

mkdir -p "${SCRIPT_DIR}/outputs/embeddings"

# (variant_name, module_relative_to_galaxy_model, model_class)
declare -a VARIANTS=(
  "base|double_train_fm_neighbors.py|ConditionalFlowMatchingModule"
  "hier|hierarchical_attention/double_train_fm_neighbors_hier_global_ins.py|HierarchicalGlobalInstrumentFlowMatchingModule"
  "hier-small|hierarchical_attention/double_train_fm_neighbors_hier_global_ins.py|HierarchicalGlobalInstrumentFlowMatchingModule"
  "single-baseline|single_encoder_ablation/model.py|SingleEncoderFlowMatchingModule"
)

for entry in "${VARIANTS[@]}"; do
    IFS='|' read -r NAME MODULE CLASS <<< "$entry"
    CKPT="${CKPT_ROOT}/${NAME}/snapshot.ckpt"
    OUT="${SCRIPT_DIR}/outputs/embeddings/${NAME}.h5"
    if [[ ! -f "${CKPT}" ]]; then
        echo "[skip] ${NAME}: missing ${CKPT}"
        continue
    fi
    echo "=== Embedding for ${NAME} ==="
    python "${SCRIPT_DIR}/prepare_combined.py" \
        --checkpoint "${CKPT}" \
        --module     "${MODULE}" \
        --model-class "${CLASS}" \
        --index-dir  "${SCRIPT_DIR}/outputs/index" \
        --out        "${OUT}" \
        --batch-size 256
done

echo "Done at $(date)"
