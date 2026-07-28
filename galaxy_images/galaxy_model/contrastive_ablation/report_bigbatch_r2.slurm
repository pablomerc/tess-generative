#!/bin/bash
#SBATCH --job-name=contr-r2-report
#SBATCH --partition=mit_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=/orcd/pool/007/pablomer/tess-generative/galaxy_images/galaxy_model/contrastive_ablation/logs/%x_%j.out
#SBATCH --error=/orcd/pool/007/pablomer/tess-generative/galaxy_images/galaxy_model/contrastive_ablation/logs/%x_%j.err
#
# Posts the arm-I batch-640 downstream-R2 report to Discord. CPU-only, ~10s of work.
#
# Submit with a dependency on the eval jobs so it runs when they are done — afterANY,
# not afterok, so a failed eval still gets reported rather than silently never arriving:
#
#   sbatch --dependency=afterany:<eval1>:<eval2> \
#     --export=ALL,DISCORD_WEBHOOK='https://...' \
#     galaxy_images/galaxy_model/contrastive_ablation/report_bigbatch_r2.slurm

set -eo pipefail

REPO_ROOT="${REPO_ROOT:-/orcd/pool/007/pablomer/tess-generative}"
CA_DIR="${REPO_ROOT}/galaxy_images/galaxy_model/contrastive_ablation"
CKPT_DIR="${CKPT_DIR:-/orcd/pool/007/pablomer/checkpoints_new/contrastive-spatial-conv1x1-BS640-DWNVAL}"

LATEST_MD="${LATEST_MD:-${CA_DIR}/RESULT_contrastive-spatial-conv1x1-BS640-DWNVAL.md}"
BESTVAL_MD="${BESTVAL_MD:-${CA_DIR}/RESULT_contrastive-spatial-conv1x1-BS640-bestval-DWNVAL.md}"

source ~/.bashrc
conda activate torchenv
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
cd "${REPO_ROOT}"

# Name the checkpoints in the report rather than assuming: read the step the run actually
# ended on, and whichever best-*.ckpt the eval auto-picked.
FINAL_STEP="$(sed -n 's/^global_step=//p' "${CKPT_DIR}/DONE" 2>/dev/null | head -1)"
FINAL_STEP="${FINAL_STEP:-unknown}"
BEST_NAME="$(basename "$(ls -t "${CKPT_DIR}"/best-*.ckpt 2>/dev/null | head -1)" 2>/dev/null || echo "best-?.ckpt")"
BEST_STEP="$(echo "${BEST_NAME}" | sed -n 's/.*step=\([0-9]*\)\.ckpt/\1/p')"
BEST_STEP="${BEST_STEP:-?}"

# 640 anchors/step, so images seen = step * 640.
imgs() { awk -v s="$1" 'BEGIN{ if (s+0>0) printf "%.1fM imgs", s*640/1e6; else printf "? imgs" }'; }

echo "[report] latest=step ${FINAL_STEP} | best-val=${BEST_NAME} (step ${BEST_STEP})"

python "${CA_DIR}/report_bigbatch_r2.py" \
    --arm "latest checkpoint — step ${FINAL_STEP} ($(imgs "${FINAL_STEP}")), the equal-optimizer-steps arm=${LATEST_MD}" \
    --arm "best-val checkpoint — step ${BEST_STEP} ($(imgs "${BEST_STEP}")), selected on val/loss like every other arm=${BESTVAL_MD}" \
    --webhook "${DISCORD_WEBHOOK:-}"

echo "Done at $(date)"
