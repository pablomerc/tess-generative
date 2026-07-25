#!/bin/bash
#SBATCH --job-name=contr-dwnval-eval
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --partition=mit_normal_gpu
#SBATCH --mem=32G
#SBATCH --output=/orcd/pool/007/pablomer/tess-generative/galaxy_images/galaxy_model/contrastive_ablation/logs/%x_%j.out
#SBATCH --error=/orcd/pool/007/pablomer/tess-generative/galaxy_images/galaxy_model/contrastive_ablation/logs/%x_%j.err
#SBATCH --open-mode=append
#
# Downstream evaluation of the contrastive DWNVAL checkpoint — ENGAGING ONLY.
#
# This half must run here: prepare_combined.py reads the downstream image stores
# (/home/pablomer/orcd/scratch/{hsc_downstream,legacy_downstream_full}) and the
# overlap index (downstream_evaluation/engaging/outputs/index/), neither of which
# exists on the AMD cluster.
#
# Protocol is byte-identical to how contrastive-spatial-conv1x1 was evaluated
# (same prepare/predict scripts, --mlp-arch aion, --seed 0, same index dir), so the
# holdout-vs-no-holdout delta is the only thing that moves.
#
# Prereq: the checkpoint from the training cluster is in
#   /orcd/pool/007/pablomer/checkpoints_new/contrastive-${RUN_TAG}/
#
# Submit:
#   sbatch --export=ALL,RUN_TAG=spatial-conv1x1-DWNVAL \
#     galaxy_images/galaxy_model/contrastive_ablation/eval_dwnval_engaging.slurm
#
# Optional env: CKPT (explicit checkpoint path), CKPT_DIR, SEEDS (default "0"),
#               DISCORD_WEBHOOK.

set -eo pipefail

REPO_ROOT="${REPO_ROOT:-/home/pablomer/orcd/pool/tess-generative}"
ENG_DIR="${REPO_ROOT}/galaxy_images/galaxy_model/downstream_evaluation/engaging"
CA_DIR="${REPO_ROOT}/galaxy_images/galaxy_model/contrastive_ablation"

RUN_TAG="${RUN_TAG:-spatial-conv1x1-DWNVAL}"
VAR="contrastive-${RUN_TAG}"
CKPT_DIR="${CKPT_DIR:-/orcd/pool/007/pablomer/checkpoints_new/${VAR}}"
# The reference arm this run is an A/B against.
REF_VAR="${REF_VAR:-contrastive-spatial-conv1x1}"
SEEDS="${SEEDS:-0}"

mkdir -p "${CA_DIR}/logs"

_ping() {
    [[ -z "${DISCORD_WEBHOOK:-}" ]] && return 0
    curl -sS -X POST -H 'Content-Type: application/json' \
        -d "{\"content\": \"$1\"}" "${DISCORD_WEBHOOK}" >/dev/null 2>&1 || true
}
_on_exit() { rc=$?; [[ $rc -ne 0 ]] && _ping "${VAR} eval ❌ failed (rc=$rc) job ${SLURM_JOB_ID:-?}"; return 0; }
trap _on_exit EXIT

# Prefer the best-val checkpoint; fall back to the rolling last.ckpt.
if [[ -z "${CKPT:-}" ]]; then
    CKPT=$(ls -t "${CKPT_DIR}"/best-*.ckpt 2>/dev/null | head -1 || true)
    [[ -z "${CKPT}" ]] && CKPT="${CKPT_DIR}/last.ckpt"
fi
if [[ ! -f "${CKPT}" ]]; then
    echo "[eval] no checkpoint found. Looked in ${CKPT_DIR} for best-*.ckpt then last.ckpt."
    echo "[eval] copy the trained checkpoint from the training cluster first, e.g.:"
    echo "  rsync -avP <amd>:/work1/jeroenaudenaert/pablomer/outputs/contrastive_ablation/${VAR}/best-*.ckpt ${CKPT_DIR}/"
    _ping "${VAR} eval ❌ no checkpoint in ${CKPT_DIR}"
    exit 1
fi
echo "[eval] checkpoint: ${CKPT}"
_ping "${VAR} 🔎 downstream eval started (ckpt=$(basename "${CKPT}")) job ${SLURM_JOB_ID:-?}"

source ~/.bashrc
conda activate torchenv
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export TORCH_BLAS_PREFER_HIPBLASLT=0     # harmless no-op on NVIDIA
export OMP_NUM_THREADS=1
export WANDB_MODE=disabled
cd "${REPO_ROOT}"

mkdir -p "${ENG_DIR}/outputs/embeddings" "${ENG_DIR}/outputs/predictions" \
         "${ENG_DIR}/outputs/plots_publication"

EMB="${ENG_DIR}/outputs/embeddings/${VAR}.h5"

# 1) Frozen embeddings on the fixed n=5,469 overlap sample: physics (encoder_galaxy)
#    and instrument (encoder_instrument), raw encoder outputs (no projection heads).
#    Same model class as the non-holdout arm, so no eval-side change is needed.
python "${ENG_DIR}/prepare_combined.py" \
    --checkpoint  "${CKPT}" \
    --module      contrastive_baseline/dual_encoder_contrastive.py \
    --model-class DualEncoderContrastiveModule \
    --index-dir   "${ENG_DIR}/outputs/index" \
    --out         "${EMB}"

# 2) MLP probes. Loop over seeds so the DWNVAL delta can be read against probe noise
#    rather than mistaken for it (SEEDS="0 1 2" gives a spread; default is seed 0 only,
#    which is what every published number used).
for S in ${SEEDS}; do
    if [[ "${S}" == "0" ]]; then
        CSV="${ENG_DIR}/outputs/predictions/predict_${VAR}__aion.csv"
    else
        CSV="${ENG_DIR}/outputs/predictions/predict_${VAR}__aion_seed${S}.csv"
    fi
    echo "[eval] probe seed=${S} -> ${CSV}"
    python "${ENG_DIR}/predict_combined.py" \
        --embeddings "${EMB}" \
        --out        "${CSV}" \
        --mlp-arch   aion \
        --seed       "${S}"
done

# 3) The actual deliverable: per-cell R² delta vs the non-holdout arm.
python "${CA_DIR}/compare_dwnval.py" \
    --holdout-csv    "${ENG_DIR}/outputs/predictions/predict_${VAR}__aion.csv" \
    --reference-csv  "${ENG_DIR}/outputs/predictions/predict_${REF_VAR}__aion.csv" \
    --out-md         "${CA_DIR}/DWNVAL_RESULT.md"

# 4) Overlay against Ours on the paper figure (same call the other arms used).
python "${ENG_DIR}/plot_unified_three_probes.py" \
    --mlp-arch aion --no-cross-predict \
    --extra-variant "${REF_VAR}:Contrastive" \
    --extra-variant "${VAR}:Contrastive (holdout)" \
    --suffix "${VAR}_vs_ours"

_ping "${VAR} ✅ eval done — see contrastive_ablation/DWNVAL_RESULT.md for the ΔR² table."
echo "Done at $(date)"
