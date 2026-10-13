#!/bin/bash
#SBATCH --job-name=jepa-eval
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --partition=mit_normal_gpu
#SBATCH --mem=32G
#SBATCH --output=/orcd/pool/007/pablomer/tess-generative/galaxy_images/galaxy_model/jepa_ablation/logs/%x_%j.out
#SBATCH --error=/orcd/pool/007/pablomer/tess-generative/galaxy_images/galaxy_model/jepa_ablation/logs/%x_%j.err
#SBATCH --open-mode=append
#
# Downstream eval for ONE trained JEPA arm (see PLAN.md):
#   1. n=5469 R² pipeline (prepare_combined -> predict_combined -> overlay plot)
#   2. paper-style survey UMAP (regenerate_umap_jepa.py, inline — GPU is here)
#   3. anomaly pipeline (contrastive_anomalies/run_pipeline.slurm, separate job)
#
#   sbatch --export=ALL,ARM=cf --job-name=jepa-eval-cf eval_jepa.slurm

set -eo pipefail

REPO_ROOT="/home/pablomer/orcd/pool/tess-generative"
GM_DIR="${REPO_ROOT}/galaxy_images/galaxy_model"
ENG_DIR="${GM_DIR}/downstream_evaluation/engaging"
JEPA_DIR="${GM_DIR}/jepa_ablation"

: "${ARM:?set ARM (hsc-resnet|hsc-vit|cf|cross)}"
VAR="jepa-${ARM}"
CKPT_DIR="/orcd/pool/007/pablomer/checkpoints_new/${VAR}"

case "${ARM}" in
  hsc-resnet) MODULE="jepa_ablation/jepa_modules.py"; CLASS="JEPAHSCModule";;
  hsc-vit)    MODULE="jepa_ablation/jepa_vit.py";     CLASS="IJEPAViTModule";;
  cf)         MODULE="jepa_ablation/jepa_modules.py"; CLASS="JEPACounterfactualModule";;
  cf-vit)     MODULE="jepa_ablation/jepa_modules.py"; CLASS="JEPACounterfactualModule";;
  cross)      MODULE="jepa_ablation/jepa_modules.py"; CLASS="JEPACrossSurveyModule";;
  ae-hsc)     MODULE="jepa_ablation/autoencoder_baseline.py"; CLASS="AEHSCModule";;
  cf-token)   MODULE="jepa_ablation/jepa_round3.py"; CLASS="CFTokenJEPAModule";;
  xjepa)      MODULE="jepa_ablation/jepa_round3.py"; CLASS="XJEPAModule";;
  multi-vit)  MODULE="jepa_ablation/jepa_vit.py";    CLASS="IJEPAViTModule";;
  *) echo "unknown ARM=${ARM}"; exit 1;;
esac

WEBHOOK="${DISCORD_WEBHOOK:-https://discord.com/api/webhooks/1500242707522519183/NqqAxuo5n0Ex4X81vMcq7gDOjrP3oAURAVZtgrZT5CramGNfBGPxuEvC2NnxwYBkfs9j}"
_ping() { curl -sS -X POST -H 'Content-Type: application/json' -d "{\"content\": \"$1\"}" "${WEBHOOK}" >/dev/null 2>&1 || true; }
_on_exit() { rc=$?; if [[ $rc -ne 0 ]]; then _ping "${VAR} eval ❌ failed (rc=$rc) job ${SLURM_JOB_ID:-?}"; fi; exit $rc; }
trap _on_exit EXIT

# Use the FINAL checkpoint, NOT best-on-val/loss.
#
# This deliberately differs from eval_contrastive_engaging.slurm. A JEPA/BYOL
# prediction loss is not a model-selection criterion: the target side is learned,
# so the loss can be driven down by making the target trivially predictable
# (collapse), and it legitimately RISES as the representation becomes richer and
# harder to predict. Empirically the hsc-vit arm's best val/loss landed at step
# 2,000 of 100,000 — selecting it would have evaluated an untrained model. Standard
# practice for this family is the final EMA weights; the best-*.ckpt files are kept
# for diagnostics only.
CKPT="${CKPT_DIR}/latest-step=${MAX_STEPS:-100000}.ckpt"
[[ -f "${CKPT}" ]] || CKPT=$(ls -t "${CKPT_DIR}"/latest-step=*.ckpt 2>/dev/null | head -1 || true)
[[ -z "${CKPT}" || ! -f "${CKPT}" ]] && CKPT="${CKPT_DIR}/last.ckpt"
if [[ ! -f "${CKPT}" ]]; then
    _ping "${VAR} eval ❌ no checkpoint found in ${CKPT_DIR}"
    exit 1
fi
echo "[eval] checkpoint: ${CKPT}"
_ping "${VAR} 🔎 downstream eval started (ckpt=$(basename "${CKPT}")) job ${SLURM_JOB_ID:-?}"

source ~/.bashrc
conda activate torchenv
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1
export WANDB_MODE=disabled
export DISCORD_WEBHOOK="${WEBHOOK}"
cd "${REPO_ROOT}"

mkdir -p "${ENG_DIR}/outputs/embeddings" "${ENG_DIR}/outputs/predictions" "${ENG_DIR}/outputs/plots_publication"

EMB="${ENG_DIR}/outputs/embeddings/${VAR}.h5"
CSV="${ENG_DIR}/outputs/predictions/predict_${VAR}__aion.csv"

# 0) Collapse check BEFORE the probe: an R² number from a collapsed encoder is not
#    interpretable, so the verdict is recorded alongside the eval rather than
#    discovered later. Must run after the conda activate above (needs torchenv).
#    Every failure path is swallowed — a missing verdict must never abort the eval
#    (note `|| true` on the assignment: under `set -e` + pipefail a grep that
#    matches nothing would otherwise kill the script).
python "${JEPA_DIR}/check_collapse.py" \
    --arm "${ARM}" --checkpoint "${CKPT}" --n 1024 \
    > "${CKPT_DIR}/collapse_check.txt" 2>&1 || echo "[warn] collapse check failed" >&2
cat "${CKPT_DIR}/collapse_check.txt" || true
COLLAPSE_SUMMARY=$(grep -oE "\-> (COLLAPSED|weak|healthy)" "${CKPT_DIR}/collapse_check.txt" 2>/dev/null | sed 's/-> //' | paste -sd'/' - || true)
_ping "${VAR} 🧪 collapse check: ${COLLAPSE_SUMMARY:-unavailable}"

# 1) Extract EMA-encoder latents on the fixed n=5469 overlap sample.
python "${ENG_DIR}/prepare_combined.py" \
    --checkpoint  "${CKPT}" \
    --module      "${MODULE}" \
    --model-class "${CLASS}" \
    --index-dir   "${ENG_DIR}/outputs/index" \
    --out         "${EMB}"

# 2) MLP probe (strict AION arch, 90/10 split, seed 0) — same as "Ours".
python "${ENG_DIR}/predict_combined.py" \
    --embeddings "${EMB}" \
    --out        "${CSV}" \
    --mlp-arch   aion \
    --seed       0

# 3) Overlay THIS arm against Ours + the flagship contrastive arm, and post.
python "${ENG_DIR}/plot_unified_three_probes.py" \
    --mlp-arch aion --no-cross-predict \
    --extra-variant "contrastive-spatial-conv1x1:Contrastive" \
    --extra-variant "${VAR}:JEPA-${ARM}" \
    --suffix "${VAR}_vs_ours"

_ping "${VAR} ✅ R² eval done — plot vs Ours/Contrastive posted above."

# 4) If ALL four JEPA arms are evaluated, render one combined comparison plot.
ALL_ARMS=(hsc-resnet hsc-vit cf cross)
missing=0
EXTRA_ARGS=(--extra-variant "contrastive-spatial-conv1x1:Contrastive")
for a in "${ALL_ARMS[@]}"; do
    c="${ENG_DIR}/outputs/predictions/predict_jepa-${a}__aion.csv"
    if [[ -f "${c}" ]]; then
        EXTRA_ARGS+=(--extra-variant "jepa-${a}:J-${a}")
    else
        missing=1
    fi
done
if [[ "${missing}" -eq 0 ]]; then
    echo "[eval] all 4 JEPA arms present — rendering combined comparison plot"
    python "${ENG_DIR}/plot_unified_three_probes.py" \
        --mlp-arch aion --no-cross-predict \
        "${EXTRA_ARGS[@]}" \
        --suffix "jepa_all_vs_ours"
    _ping "🧩 All 4 JEPA arms evaluated — combined R² comparison vs Ours/Contrastive posted above."
fi

# 5) Paper-style survey UMAP (inline — this job has the GPU).
python -m galaxy_images.galaxy_model.jepa_ablation.regenerate_umap_jepa \
    --arm "${ARM}" \
    --checkpoint "${CKPT}" \
    --num-anchors 2000 \
    --tag "${VAR}-j${SLURM_JOB_ID:-0}" \
    || _ping "${VAR} ⚠️ UMAP regeneration failed (non-fatal, continuing)"

# 6) Anomaly pipeline over the 103,741 anchors (separate pg_mki_aryeh job; its
#    relative logs/ path resolves against the submit cwd, so cd there first).
cd "${GM_DIR}/contrastive_anomalies"
sbatch --job-name="jepa-anom-${ARM}" \
       --export=ALL,CKPT="${CKPT}",TAG="${VAR}" \
       run_pipeline.slurm \
    && _ping "${VAR} 📤 anomaly pipeline submitted" \
    || _ping "${VAR} ⚠️ failed to submit anomaly pipeline"

echo "Done at $(date)"
