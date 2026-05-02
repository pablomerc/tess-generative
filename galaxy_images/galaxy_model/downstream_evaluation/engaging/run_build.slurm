#!/bin/bash
#SBATCH --job-name=engaging-build
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=1:00:00
# NOTE(engaging): pick the right --partition for the engaging cluster.
# On the AMD cluster this was --partition=mi2104x; on engaging it's likely
# a CPU partition since this stage is pure CPU + a few GB of RAM.
##SBATCH --partition=<engaging-cpu-partition>
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --open-mode=append

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

source ~/.bashrc
# NOTE(engaging): replace torchenv with whatever conda env you set up there.
conda activate torchenv
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

cd "${REPO_ROOT}"

python "${SCRIPT_DIR}/build_overlap_dataset.py" \
    --hsc-dir    /work1/jeroenaudenaert/pablomer/data/hsc_downstream \
    --legacy-dir /work1/jeroenaudenaert/pablomer/data/legacy_downstream_quick \
    --out-dir    "${SCRIPT_DIR}/outputs/index" \
    --max-per-subset 16000 \
    --match-radius-arcsec 1.0 \
    --seed 42

echo "Done at $(date)"
