#!/bin/bash
#SBATCH --job-name=jepa-smoke
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
# End-to-end smoke test for the JEPA ablation (PLAN.md): 30-step micro-runs of
# all four arms (memmap data path, offline wandb), then the downstream
# integration surfaces on the micro-checkpoints:
#   * prepare_combined on the `cross` ckpt  (duck-typed property alias, e1==e2)
#   * prepare_combined on the `hsc-vit` ckpt (_MeanPoolViT wrapper, 192-d)
#   * contrastive_anomalies/encode_latents --limit on the `cf` ckpt (class sniff)
#   * regenerate_umap_jepa on the `cf` ckpt (plotter integration)
#
#   sbatch smoke_test.slurm

set -eo pipefail

REPO_ROOT="/home/pablomer/orcd/pool/tess-generative"
GM_DIR="${REPO_ROOT}/galaxy_images/galaxy_model"
JEPA_DIR="${GM_DIR}/jepa_ablation"
ENG_DIR="${GM_DIR}/downstream_evaluation/engaging"
SMOKE_DIR="/orcd/data/mki_aryeh/001/pablomer/jepa_ablation/smoke"

WEBHOOK="${DISCORD_WEBHOOK:-https://discord.com/api/webhooks/1500242707522519183/NqqAxuo5n0Ex4X81vMcq7gDOjrP3oAURAVZtgrZT5CramGNfBGPxuEvC2NnxwYBkfs9j}"
_ping() { curl -sS -X POST -H 'Content-Type: application/json' -d "{\"content\": \"$1\"}" "${WEBHOOK}" >/dev/null 2>&1 || true; }
STAGE="startup"
_on_exit() {
    rc=$?
    if [[ $rc -eq 0 ]]; then
        _ping "jepa-smoke ✅ all stages passed (job ${SLURM_JOB_ID:-?}) — ready to submit the real runs"
    else
        _ping "jepa-smoke ❌ failed in stage '${STAGE}' (rc=$rc, job ${SLURM_JOB_ID:-?})"
    fi
}
trap _on_exit EXIT

_ping "jepa-smoke 🚀 start (job ${SLURM_JOB_ID:-?}, host ${SLURMD_NODENAME:-?})"

source ~/.bashrc
conda activate torchenv
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export WANDB_MODE=offline
export DISCORD_WEBHOOK="${WEBHOOK}"
cd "${REPO_ROOT}"
mkdir -p "${SMOKE_DIR}"

# ---- 1) 30-step micro-run per arm -----------------------------------------
for ARM in cf cross hsc-resnet hsc-vit; do
    STAGE="train-${ARM}"
    echo "=== [smoke] training arm=${ARM} ==="
    rm -rf "${SMOKE_DIR}/${ARM}"
    export WANDB_DIR="${SMOKE_DIR}/${ARM}"
    mkdir -p "${WANDB_DIR}"
    python "${JEPA_DIR}/train_jepa.py" \
        --arm "${ARM}" \
        --max-steps 30 \
        --batch-size 8 \
        --num-workers 2 \
        --val-check-interval 20 \
        --limit-val-batches 2 \
        --load-into-ram 0 \
        --umap-pairs 64 \
        --ckpt-dir "${SMOKE_DIR}/${ARM}" \
        --run-name "smoke-${ARM}" \
        --wandb-project galaxy-jepa-smoke
    [[ -f "${SMOKE_DIR}/${ARM}/DONE" ]] || { echo "[smoke] ${ARM}: no DONE marker"; exit 1; }
done

_pick_ckpt() {  # best if present, else last
    local d="$1"
    local c
    c=$(ls -t "${d}"/best-*.ckpt 2>/dev/null | head -1 || true)
    [[ -z "${c}" ]] && c="${d}/last.ckpt"
    echo "${c}"
}

# ---- 2) prepare_combined on cross (alias path) and hsc-vit (wrapper path) ---
for ARM in cross hsc-vit; do
    STAGE="prepare-${ARM}"
    case "${ARM}" in
      hsc-vit) MODULE="jepa_ablation/jepa_vit.py";     CLASS="IJEPAViTModule";;
      cross)   MODULE="jepa_ablation/jepa_modules.py"; CLASS="JEPACrossSurveyModule";;
    esac
    echo "=== [smoke] prepare_combined arm=${ARM} ==="
    python "${ENG_DIR}/prepare_combined.py" \
        --checkpoint  "$(_pick_ckpt "${SMOKE_DIR}/${ARM}")" \
        --module      "${MODULE}" \
        --model-class "${CLASS}" \
        --index-dir   "${ENG_DIR}/outputs/index" \
        --out         "${SMOKE_DIR}/${ARM}/smoke_embeddings.h5"
    python - "$SMOKE_DIR/${ARM}/smoke_embeddings.h5" <<'EOF'
import sys, h5py
with h5py.File(sys.argv[1]) as f:
    e1 = f["overlap/hsc_e1"]; e2 = f["overlap/hsc_e2"]
    print(f"[smoke-check] overlap hsc_e1 {e1.shape} hsc_e2 {e2.shape}")
    assert e1.shape[0] > 5000 and e1.shape == e2.shape
EOF
done

# ---- 3) anomaly-pipeline loader on the cf ckpt (patched class sniff) --------
STAGE="encode-latents-cf"
echo "=== [smoke] encode_latents --limit 256 on cf ==="
python "${GM_DIR}/contrastive_anomalies/encode_latents.py" \
    --ckpt "$(_pick_ckpt "${SMOKE_DIR}/cf")" \
    --tag smoke-jepa-cf \
    --out-dir "${SMOKE_DIR}/anomaly_smoke" \
    --limit 256

# ---- 4) paper-style UMAP on the cf ckpt -------------------------------------
STAGE="umap-cf"
echo "=== [smoke] regenerate_umap_jepa on cf ==="
python -m galaxy_images.galaxy_model.jepa_ablation.regenerate_umap_jepa \
    --arm cf \
    --checkpoint "$(_pick_ckpt "${SMOKE_DIR}/cf")" \
    --num-anchors 128 \
    --num-workers 2 \
    --tag "smoke-j${SLURM_JOB_ID:-0}"

STAGE="done"
echo "=== [smoke] ALL STAGES PASSED ==="
