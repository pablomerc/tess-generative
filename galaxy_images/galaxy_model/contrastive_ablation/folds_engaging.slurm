#!/bin/bash
#SBATCH --job-name=contr-folds
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --partition=mit_normal
#SBATCH --mem=32G
#SBATCH --output=/orcd/pool/007/pablomer/tess-generative/galaxy_images/galaxy_model/contrastive_ablation/logs/%x_%j.out
#SBATCH --error=/orcd/pool/007/pablomer/tess-generative/galaxy_images/galaxy_model/contrastive_ablation/logs/%x_%j.err
#SBATCH --open-mode=append
#
# 5-fold error bars for the Ours-vs-contrastive R² comparison — ENGAGING.
#
# The published numbers come from a single 90/10 split with n_test ~= 470-550, so
# several of the gaps we want to quote in the rebuttal (e.g. redshift 0.740 vs 0.801)
# are of the same order as the split noise. This runs a disjoint 5-way partition
# (every galaxy tested exactly once) for the four latent variants that make up the
# dim-matched comparison, then aggregates to mean +- std per cell.
#
# CPU partition on purpose: these are 2-layer MLPs on <=5,469 rows, so a GPU would
# spend its time idle and the CPU queue is shorter.
#
#   sbatch galaxy_images/galaxy_model/contrastive_ablation/folds_engaging.slurm
#
# Optional env: VARIANTS (space-separated embedding basenames), N_FOLDS (default 5).

set -eo pipefail

REPO_ROOT="${REPO_ROOT:-/home/pablomer/orcd/pool/tess-generative}"
ENG_DIR="${REPO_ROOT}/galaxy_images/galaxy_model/downstream_evaluation/engaging"
CA_DIR="${REPO_ROOT}/galaxy_images/galaxy_model/contrastive_ablation"

N_FOLDS="${N_FOLDS:-5}"
# Ours flat / Ours pooled / contrastive flat / contrastive pooled — the four cells of
# the dim-matched comparison in PLAN.md §1.
VARIANTS="${VARIANTS:-base base-meanpool contrastive-spatial-conv1x1 contrastive-spatial-conv1x1-meanpool}"

mkdir -p "${CA_DIR}/logs" "${ENG_DIR}/outputs/predictions/folds"

_ping() {
    [[ -z "${DISCORD_WEBHOOK:-}" ]] && return 0
    curl -sS -X POST -H 'Content-Type: application/json' \
        -d "{\"content\": \"$1\"}" "${DISCORD_WEBHOOK}" >/dev/null 2>&1 || true
}

source ~/.bashrc
conda activate torchenv
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MPLCONFIGDIR=/tmp/matplotlib
export NUMBA_CACHE_DIR=/tmp/numba_cache
export WANDB_MODE=disabled
mkdir -p /tmp/matplotlib /tmp/numba_cache
cd "${ENG_DIR}"

echo "[folds] ${N_FOLDS}-fold CV over: ${VARIANTS}"
_ping "📊 5-fold R² error bars started (job ${SLURM_JOB_ID:-?})"

for VAR in ${VARIANTS}; do
    EMB="outputs/embeddings/${VAR}.h5"
    if [[ ! -f "${EMB}" ]]; then
        echo "[folds] SKIP ${VAR} — no embeddings at ${EMB}"
        continue
    fi
    for (( F=0; F<N_FOLDS; F++ )); do
        OUT="outputs/predictions/folds/predict_${VAR}__aion_f${F}of${N_FOLDS}.csv"
        if [[ -f "${OUT}" ]]; then
            echo "[folds] ${VAR} fold ${F}: exists, skipping"
            continue
        fi
        echo "[folds] ${VAR} fold ${F}/${N_FOLDS}"
        python predict_combined.py \
            --embeddings "${EMB}" \
            --out        "${OUT}" \
            --mlp-arch   aion \
            --seed       0 \
            --n-folds    "${N_FOLDS}" \
            --fold       "${F}" \
            --device     cpu
    done
done

# Aggregate every fold CSV into mean +- std per (variant, task_family, target, latent).
python "${CA_DIR}/aggregate_folds.py" \
    --folds-dir "outputs/predictions/folds" \
    --n-folds   "${N_FOLDS}" \
    --out-csv   "outputs/predictions/folds_r2_summary.csv" \
    --out-md    "${CA_DIR}/RESULT_folds.md"

_ping "✅ 5-fold R² error bars done — see contrastive_ablation/RESULT_folds.md"
echo "Done at $(date)"
