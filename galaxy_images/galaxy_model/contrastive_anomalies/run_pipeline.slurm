#!/bin/bash
#SBATCH --job-name=contr-anom
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --partition=pg_mki_aryeh
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --open-mode=append
#
# Anomaly detection on the contrastive dual-encoder's physics and instrument
# latent spaces, over the 103,741-anchor efficient_neighs_48 corpus.
#
# All four stages in one job — the whole thing is ~1 h, so chaining four jobs
# would spend more time queueing than computing.
#
#   sbatch run_pipeline.slurm
#
# Other arms of the 2x2 contrastive sweep:
#   CKPT=/orcd/pool/007/pablomer/checkpoints_new/contrastive-random-conv1x1/best-epoch=59-step=91000.ckpt \
#   TAG=random-conv1x1 sbatch run_pipeline.slurm
#
# Reproduce the legacy AMD scoring protocol (self-matching kNN, no standardisation):
#   FIT_EXTRA="--knn-allow-self --no-standardize" TAG=spatial-conv1x1-legacyproto sbatch run_pipeline.slurm

set -eo pipefail

REPO=/orcd/pool/007/pablomer/tess-generative
SRC="${REPO}/galaxy_images/galaxy_model/contrastive_anomalies"
TORCHENV_PY=/orcd/home/002/pablomer/miniforge3/envs/torchenv/bin/python

: "${CKPT:=/orcd/pool/007/pablomer/checkpoints_new/contrastive-spatial-conv1x1/best-epoch=53-step=83000.ckpt}"
: "${TAG:=spatial-conv1x1}"
: "${DATA_DIR:=/orcd/pool/007/pablomer/efficient_neighs_48}"
: "${OUT_DIR:=/orcd/pool/007/pablomer/outputs/contrastive_anomalies}"
: "${LIMIT:=}"           # empty = all 103,741 anchors
: "${NSF_EPOCHS:=50}"
: "${FIT_EXTRA:=}"
WEBHOOK="${DISCORD_WEBHOOK:-https://discord.com/api/webhooks/1500242707522519183/NqqAxuo5n0Ex4X81vMcq7gDOjrP3oAURAVZtgrZT5CramGNfBGPxuEvC2NnxwYBkfs9j}"

_ping() { curl -sS -X POST -H 'Content-Type: application/json' \
          -d "{\"content\": \"$(printf '%s' "$1" | sed 's/"/\\"/g')\"}" "${WEBHOOK}" >/dev/null 2>&1 || true; }

STAGE="startup"
cleanup() {
  local rc=$?
  if [[ ${rc} -eq 0 ]]; then
    _ping "contr-anom ✅ done · tag=${TAG} · job ${SLURM_JOB_ID:-?} · results in \`${OUT_DIR}\`"
  else
    _ping "contr-anom ❌ failed in stage '${STAGE}' (exit ${rc}) · tag=${TAG} · job ${SLURM_JOB_ID:-?}"
  fi
}
trap cleanup EXIT

_ping "contr-anom 🚀 start · tag=${TAG} · job ${SLURM_JOB_ID:-?} · host ${SLURMD_NODENAME:-?} · physics+instrument anomaly detection on ~104k anchors"

source ~/.bashrc
conda activate torchenv || true

export PYTHONPATH="${REPO}:${PYTHONPATH}"
# One thread per BLAS op; the parallelism we want is joblib's over IForest trees.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
cd "${SRC}"
mkdir -p logs "${OUT_DIR}"

LATENTS="${OUT_DIR}/latents_${TAG}.h5"
SCORES="${OUT_DIR}/scores_${TAG}.h5"
LIMIT_ARG=""
[[ -n "${LIMIT}" ]] && LIMIT_ARG="--limit ${LIMIT}"

echo "=== config ==="
echo "  ckpt    : ${CKPT}"
echo "  tag     : ${TAG}"
echo "  data    : ${DATA_DIR}"
echo "  out     : ${OUT_DIR}"
"${TORCHENV_PY}" -c "import torch;print('  torch   :',torch.__version__,'cuda',torch.cuda.is_available())"

STAGE="encode"
if [[ -f "${LATENTS}" ]]; then
  echo "=== [1/4] encode — reusing existing ${LATENTS} ==="
else
  echo "=== [1/4] encode ==="
  srun "${TORCHENV_PY}" -u encode_latents.py \
      --ckpt "${CKPT}" --tag "${TAG}" --data-dir "${DATA_DIR}" --out-dir "${OUT_DIR}" ${LIMIT_ARG}
fi

STAGE="fit"
echo "=== [2/4] fit + score ==="
srun "${TORCHENV_PY}" -u fit_and_score.py \
    --latents "${LATENTS}" --tag "${TAG}" --out-dir "${OUT_DIR}" \
    --nsf-epochs "${NSF_EPOCHS}" ${FIT_EXTRA}

STAGE="evaluate"
echo "=== [3/4] evaluate ==="
srun "${TORCHENV_PY}" -u evaluate.py --scores "${SCORES}" --out-dir "${OUT_DIR}"

STAGE="plot"
echo "=== [4/4] figures ==="
srun "${TORCHENV_PY}" -u plot_anomalies.py \
    --scores "${SCORES}" --data-dir "${DATA_DIR}" --out-dir "${OUT_DIR}/figures_${TAG}"

STAGE="report"
echo "=== done $(date) ==="
ls -lh "${OUT_DIR}" "${OUT_DIR}/figures_${TAG}"

# Post the headline table + the joint figure so the result lands in Discord, not just on disk.
SUMMARY="${OUT_DIR}/summary_${TAG}.md"
if [[ -f "${SUMMARY}" ]]; then
  curl -sS -F "payload_json={\"content\": \"contr-anom 📊 ${TAG} — lens-retrieval summary\"}" \
       -F "file1=@${SUMMARY}" "${WEBHOOK}" >/dev/null 2>&1 || true
fi
JOINT="${OUT_DIR}/figures_${TAG}/joint_top12_flow_${TAG}.png"
ROC="${OUT_DIR}/figures_${TAG}/roc_lens_${TAG}.png"
[[ -f "${JOINT}" ]] && curl -sS -F "file1=@${JOINT}" "${WEBHOOK}" >/dev/null 2>&1 || true
[[ -f "${ROC}" ]]   && curl -sS -F "file1=@${ROC}"   "${WEBHOOK}" >/dev/null 2>&1 || true
