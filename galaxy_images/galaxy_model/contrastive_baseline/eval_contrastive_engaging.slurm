#!/bin/bash
#SBATCH --job-name=contr-eval
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --partition=mit_normal_gpu
#SBATCH --mem=32G
#SBATCH --output=/orcd/pool/007/pablomer/tess-generative/galaxy_images/galaxy_model/contrastive_baseline/logs/%x_%j.out
#SBATCH --error=/orcd/pool/007/pablomer/tess-generative/galaxy_images/galaxy_model/contrastive_baseline/logs/%x_%j.err
#SBATCH --open-mode=append
#
# Downstream eval for ONE trained contrastive checkpoint, matching the engaging
# n=5469 pipeline exactly (overlap.parquet sample, 90/10 MLP probe), then overlays
# the contrastive variant on the paper's "Ours" R² figure and posts it to Discord.
#
#   sbatch --export=ALL,RUN_TAG=spatial-avg eval_contrastive_engaging.slurm

set -eo pipefail

REPO_ROOT="/home/pablomer/orcd/pool/tess-generative"
ENG_DIR="${REPO_ROOT}/galaxy_images/galaxy_model/downstream_evaluation/engaging"

: "${RUN_TAG:?set RUN_TAG (e.g. spatial-avg)}"
VAR="contrastive-${RUN_TAG}"
CKPT_DIR="/orcd/pool/007/pablomer/checkpoints_new/${VAR}"

WEBHOOK="${DISCORD_WEBHOOK:-https://discord.com/api/webhooks/1500242707522519183/NqqAxuo5n0Ex4X81vMcq7gDOjrP3oAURAVZtgrZT5CramGNfBGPxuEvC2NnxwYBkfs9j}"
_ping() { curl -sS -X POST -H 'Content-Type: application/json' -d "{\"content\": \"$1\"}" "${WEBHOOK}" >/dev/null 2>&1 || true; }
_on_exit() { rc=$?; [[ $rc -ne 0 ]] && _ping "${VAR} eval ❌ failed (rc=$rc) job ${SLURM_JOB_ID:-?}"; }
trap _on_exit EXIT

# Prefer the best-val checkpoint; fall back to the rolling last.ckpt.
CKPT=$(ls -t "${CKPT_DIR}"/best-*.ckpt 2>/dev/null | head -1 || true)
[[ -z "${CKPT}" ]] && CKPT="${CKPT_DIR}/last.ckpt"
if [[ ! -f "${CKPT}" ]]; then
    _ping "${VAR} eval ❌ no checkpoint found in ${CKPT_DIR}"
    exit 1
fi
echo "[eval] checkpoint: ${CKPT}"
_ping "${VAR} 🔎 downstream eval started (ckpt=$(basename "${CKPT}")) job ${SLURM_JOB_ID:-?}"

source ~/.bashrc
conda activate torchenv
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export TORCH_BLAS_PREFER_HIPBLASLT=0
export OMP_NUM_THREADS=1
export WANDB_MODE=disabled
cd "${REPO_ROOT}"

mkdir -p "${ENG_DIR}/outputs/embeddings" "${ENG_DIR}/outputs/predictions" "${ENG_DIR}/outputs/plots_publication"

EMB="${ENG_DIR}/outputs/embeddings/${VAR}.h5"
CSV="${ENG_DIR}/outputs/predictions/predict_${VAR}__aion.csv"

# 1) Extract physics (encoder_galaxy) + instrument (encoder_instrument) latents on
#    the fixed n=5469 overlap sample.
python "${ENG_DIR}/prepare_combined.py" \
    --checkpoint  "${CKPT}" \
    --module      contrastive_baseline/dual_encoder_contrastive.py \
    --model-class DualEncoderContrastiveModule \
    --index-dir   "${ENG_DIR}/outputs/index" \
    --out         "${EMB}"

# 2) MLP probe (strict AION arch, 90/10 split, seed 0) — same as "Ours".
python "${ENG_DIR}/predict_combined.py" \
    --embeddings "${EMB}" \
    --out        "${CSV}" \
    --mlp-arch   aion \
    --seed       0

# 3) Overlay THIS variant against Ours (physics flat + instrument pooled) and post.
python "${ENG_DIR}/plot_unified_three_probes.py" \
    --mlp-arch aion --no-cross-predict \
    --extra-variant "${VAR}:Contrastive" \
    --suffix "${VAR}_vs_ours"

_ping "${VAR} ✅ downstream eval done — R² plot vs Ours posted above."

# 4) If ALL four variants are now evaluated, render one combined comparison plot.
ALL_TAGS=(spatial-avg spatial-conv1x1 random-avg random-conv1x1)
missing=0
EXTRA_ARGS=()
for t in "${ALL_TAGS[@]}"; do
    c="${ENG_DIR}/outputs/predictions/predict_contrastive-${t}__aion.csv"
    if [[ -f "${c}" ]]; then
        EXTRA_ARGS+=(--extra-variant "contrastive-${t}:C-${t}")
    else
        missing=1
    fi
done
if [[ "${missing}" -eq 0 ]]; then
    echo "[eval] all 4 variants present — rendering combined comparison plot"
    python "${ENG_DIR}/plot_unified_three_probes.py" \
        --mlp-arch aion --no-cross-predict \
        "${EXTRA_ARGS[@]}" \
        --suffix "contrastive_all_vs_ours"
    _ping "🧩 All 4 contrastive variants evaluated — combined R² comparison vs Ours posted above."
fi

echo "Done at $(date)"
