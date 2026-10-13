#!/bin/bash
#SBATCH --job-name=jepa-smoke-r3
#SBATCH --partition=pg_mki_aryeh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/orcd/pool/007/pablomer/tess-generative/galaxy_images/galaxy_model/jepa_ablation/logs/%x_%j.out
#SBATCH --error=/orcd/pool/007/pablomer/tess-generative/galaxy_images/galaxy_model/jepa_ablation/logs/%x_%j.err
#SBATCH --open-mode=append
#
# Round-3 smoke (PLAN.md §Round 3): 30-step micro-runs of cf-token / xjepa /
# multi-vit, then the riskiest eval surfaces on the micro-checkpoints:
#   * prepare_combined on xjepa (NEW class; asymmetric e1=128-d / e2=64-d widths)
#   * check_collapse on cf-token (round-3 loader path + marker guards)
# ON SUCCESS this job SUBMITS THE THREE REAL 100k-STEP RUNS.

set -eo pipefail

REPO_ROOT="/home/pablomer/orcd/pool/tess-generative"
GM_DIR="${REPO_ROOT}/galaxy_images/galaxy_model"
JEPA_DIR="${GM_DIR}/jepa_ablation"
ENG_DIR="${GM_DIR}/downstream_evaluation/engaging"
SMOKE_DIR="/orcd/data/mki_aryeh/001/pablomer/jepa_ablation/smoke_r3"

WEBHOOK="${DISCORD_WEBHOOK:-https://discord.com/api/webhooks/1500242707522519183/NqqAxuo5n0Ex4X81vMcq7gDOjrP3oAURAVZtgrZT5CramGNfBGPxuEvC2NnxwYBkfs9j}"
_ping() { curl -sS -X POST -H 'Content-Type: application/json' -d "{\"content\": \"$1\"}" "${WEBHOOK}" >/dev/null 2>&1 || true; }
STAGE="startup"
_on_exit() {
    rc=$?
    if [[ $rc -ne 0 ]]; then _ping "jepa-smoke-r3 ❌ failed in stage '${STAGE}' (rc=$rc, job ${SLURM_JOB_ID:-?}) — real runs NOT submitted"; fi
    exit $rc
}
trap _on_exit EXIT

_ping "jepa-smoke-r3 🚀 start (job ${SLURM_JOB_ID:-?})"

source ~/.bashrc
conda activate torchenv
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export WANDB_MODE=offline
export DISCORD_WEBHOOK="${WEBHOOK}"
cd "${REPO_ROOT}"
mkdir -p "${SMOKE_DIR}"

for ARM in cf-token xjepa multi-vit; do
    STAGE="train-${ARM}"
    echo "=== [smoke-r3] training arm=${ARM} ==="
    rm -rf "${SMOKE_DIR}/${ARM}"
    export WANDB_DIR="${SMOKE_DIR}/${ARM}"
    mkdir -p "${WANDB_DIR}"
    python "${JEPA_DIR}/train_jepa.py" \
        --arm "${ARM}" --max-steps 30 --batch-size 8 --num-workers 2 \
        --val-check-interval 20 --limit-val-batches 2 --load-into-ram 0 \
        --umap-pairs 64 --ckpt-dir "${SMOKE_DIR}/${ARM}" \
        --run-name "smoke-r3-${ARM}" --wandb-project galaxy-jepa-smoke
    [[ -f "${SMOKE_DIR}/${ARM}/DONE" ]] || { echo "[smoke-r3] ${ARM}: no DONE"; exit 1; }
done

STAGE="prepare-xjepa"
echo "=== [smoke-r3] prepare_combined on xjepa (asymmetric widths) ==="
python "${ENG_DIR}/prepare_combined.py" \
    --checkpoint  "${SMOKE_DIR}/xjepa/last.ckpt" \
    --module      jepa_ablation/jepa_round3.py \
    --model-class XJEPAModule \
    --index-dir   "${ENG_DIR}/outputs/index" \
    --out         "${SMOKE_DIR}/xjepa/smoke_embeddings.h5"
python - "${SMOKE_DIR}/xjepa/smoke_embeddings.h5" <<'EOF'
import sys, h5py
with h5py.File(sys.argv[1]) as f:
    e1, e2 = f["overlap/hsc_e1"], f["overlap/hsc_e2"]
    print(f"[smoke-r3] xjepa overlap e1 {e1.shape} e2 {e2.shape}")
    assert e1.shape == (5469, 128) and e2.shape == (5469, 64), "subspace widths wrong"
EOF

STAGE="collapse-cf-token"
echo "=== [smoke-r3] check_collapse on cf-token ==="
python "${JEPA_DIR}/check_collapse.py" \
    --arm cf-token --checkpoint "${SMOKE_DIR}/cf-token/last.ckpt" --n 256

STAGE="submit-real-runs"
echo "=== [smoke-r3] all stages passed — submitting the real runs ==="
for ARM in cf-token xjepa multi-vit; do
    JID=$(sbatch --parsable --job-name="jepa-${ARM}" \
                 --export=ALL,ARM=${ARM},MAX_STEPS=100000 \
                 "${JEPA_DIR}/train_jepa.slurm")
    echo "submitted jepa-${ARM}: ${JID}"
done
_ping "jepa-smoke-r3 ✅ passed — submitted cf-token / xjepa / multi-vit (100k steps each)"
echo "Done at $(date)"
