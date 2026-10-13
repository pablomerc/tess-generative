#!/bin/bash
#SBATCH --job-name=jepa-recheck
#SBATCH --partition=pg_mki_aryeh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --output=/orcd/pool/007/pablomer/tess-generative/galaxy_images/galaxy_model/jepa_ablation/logs/%x_%j.out
#SBATCH --error=/orcd/pool/007/pablomer/tess-generative/galaxy_images/galaxy_model/jepa_ablation/logs/%x_%j.err
#SBATCH --open-mode=append
#
# Re-run check_collapse.py for every finished arm with the corrected verdict logic
# (eff_rank-primary + mean-centered spread). The first version judged on rel_std
# alone and flagged the `cross` arm "weak" when it was merely offset-dominated.

set -eo pipefail
REPO_ROOT="/home/pablomer/orcd/pool/tess-generative"
JEPA_DIR="${REPO_ROOT}/galaxy_images/galaxy_model/jepa_ablation"

WEBHOOK="${DISCORD_WEBHOOK:-https://discord.com/api/webhooks/1500242707522519183/NqqAxuo5n0Ex4X81vMcq7gDOjrP3oAURAVZtgrZT5CramGNfBGPxuEvC2NnxwYBkfs9j}"
_ping() { curl -sS -X POST -H 'Content-Type: application/json' -d "{\"content\": \"$1\"}" "${WEBHOOK}" >/dev/null 2>&1 || true; }

source ~/.bashrc
conda activate torchenv
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1
cd "${REPO_ROOT}"

SUMMARY=""
for ARM in hsc-resnet hsc-vit cf cross; do
    D="/orcd/pool/007/pablomer/checkpoints_new/jepa-${ARM}"
    CKPT="${D}/latest-step=100000.ckpt"
    [[ -f "${CKPT}" ]] || CKPT="${D}/last.ckpt"
    [[ -f "${CKPT}" ]] || { echo "[skip] ${ARM}: no checkpoint"; continue; }
    echo "=== ${ARM} ==="
    python "${JEPA_DIR}/check_collapse.py" --arm "${ARM}" --checkpoint "${CKPT}" --n 1024 \
        2>&1 | tee "${D}/collapse_check.txt" | grep -E "physics/e1|instrument/e2" || true
    v=$(grep -oE "\-> [A-Za-z]+" "${D}/collapse_check.txt" 2>/dev/null | sed 's/-> //' | paste -sd'/' - || true)
    SUMMARY="${SUMMARY}${ARM}=${v:-?}  "
done

_ping "jepa 🧪 collapse re-check (corrected verdict logic): ${SUMMARY}"
echo "Done at $(date)"
