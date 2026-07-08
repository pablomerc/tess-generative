#!/bin/bash
#SBATCH --job-name=engaging-overlap-check
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0:20:00
#SBATCH --partition=mit_normal
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --open-mode=append

set -eo pipefail

SCRIPT_DIR="/home/pablomer/orcd/pool/tess-generative/galaxy_images/galaxy_model/downstream_evaluation/engaging"
REPO_ROOT="/home/pablomer/orcd/pool/tess-generative"

source ~/.bashrc
conda activate torchenv
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

DISCORD_WEBHOOK="${DISCORD_WEBHOOK:-https://discord.com/api/webhooks/1500242707522519183/NqqAxuo5n0Ex4X81vMcq7gDOjrP3oAURAVZtgrZT5CramGNfBGPxuEvC2NnxwYBkfs9j}"

cd "${REPO_ROOT}"

python "${SCRIPT_DIR}/check_overlap_visual.py" \
    --index   "${SCRIPT_DIR}/outputs/index/overlap.parquet" \
    --n       16 \
    --seed    0 \
    --out     "${SCRIPT_DIR}/outputs/plots/overlap_check.png" \
    --webhook "${DISCORD_WEBHOOK}" \
    --message "engaging downstream — overlap visual check (HSC | Legacy)"

echo "Done at $(date)"
