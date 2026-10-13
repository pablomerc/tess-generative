#!/bin/bash
#SBATCH --job-name=rebut-concat
#SBATCH --partition=mit_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --output=/orcd/pool/007/pablomer/tess-generative/galaxy_images/galaxy_model/downstream_evaluation/rebuttal_experiments/logs/%x_%A_%a.out
#SBATCH --error=/orcd/pool/007/pablomer/tess-generative/galaxy_images/galaxy_model/downstream_evaluation/rebuttal_experiments/logs/%x_%A_%a.err
#SBATCH --open-mode=append
#
# Instrument-property probes on derived latent feature sets (rebuttal experiment).
#
# CPU partition on purpose: these are 2-layer MLPs on <=5,469 rows, so a GPU would
# spend its time idle and the CPU queue is shorter. Same reasoning (and the same
# --device cpu) as contrastive_ablation/folds_engaging.slurm.
#
# Runs the UNMODIFIED ../engaging/predict_combined.py against the derived H5s built
# by build_derived_h5.py, with the exact flags of the published 5-fold run so the
# fold membership is identical and the new columns are cell-comparable:
#
#     --mlp-arch aion --seed 0 --n-folds 5 --fold F --device cpu   (max_epochs=50)
#
# Job array over (variant x fold). FOLDS is read from the environment so the same
# script serves the quick fold-0 pass and the later folds-1..4 pass:
#
#     sbatch --array=0-9        --export=ALL,FOLDS="0"         run_folds.slurm
#     sbatch --array=0-39%10    --export=ALL,FOLDS="1 2 3 4"   run_folds.slurm
#
# Resumable: a task whose output CSV already exists exits immediately.

set -uo pipefail

REPO_ROOT=/orcd/pool/007/pablomer/tess-generative
HERE="${REPO_ROOT}/galaxy_images/galaxy_model/downstream_evaluation/rebuttal_experiments"
PREDICT="${REPO_ROOT}/galaxy_images/galaxy_model/downstream_evaluation/engaging/predict_combined.py"
EMB_DIR="${HERE}/outputs/embeddings"
OUT_DIR="${HERE}/outputs/predictions/folds"

N_FOLDS=5
FOLDS="${FOLDS:-0}"
VARIANTS="${VARIANTS:-base-e1e2-256 base-e1e2m-hsc base-e1e2m-leg base-e2m-hsc base-e2m-leg contr-e1e2-256 contr-e1e2m-hsc contr-e1e2m-leg contr-e2m-hsc contr-e2m-leg}"

DISCORD_WEBHOOK="${DISCORD_WEBHOOK:-}"
_ping() {
  [ -n "${DISCORD_WEBHOOK}" ] || return 0
  curl -s -X POST -H "Content-Type: application/json" \
       -d "$(python3 -c 'import json,sys; print(json.dumps({"content": sys.argv[1]}))' "$1")" \
       "${DISCORD_WEBHOOK}" >/dev/null 2>&1 || true
}

# ---- resolve this array task to (variant, fold) -----------------------------
mapfile -t TASKS < <(for v in ${VARIANTS}; do for f in ${FOLDS}; do echo "${v} ${f}"; done; done)
IDX="${SLURM_ARRAY_TASK_ID:-0}"
if [ "${IDX}" -ge "${#TASKS[@]}" ]; then
  echo "[skip] array index ${IDX} >= ${#TASKS[@]} tasks"
  exit 0
fi
read -r VARIANT FOLD <<< "${TASKS[$IDX]}"

EMB="${EMB_DIR}/${VARIANT}.h5"
OUT="${OUT_DIR}/predict_${VARIANT}__aion_f${FOLD}of${N_FOLDS}.csv"

echo "[task ${IDX}] variant=${VARIANT} fold=${FOLD}/${N_FOLDS}"
echo "            emb=${EMB}"
echo "            out=${OUT}"

if [ -s "${OUT}" ]; then
  echo "[skip] ${OUT} already exists"
  exit 0
fi
if [ ! -s "${EMB}" ]; then
  _ping "**[rebuttal concat] MISSING EMBEDDINGS** \`${VARIANT}\` — expected \`${EMB}\`"
  echo "[fatal] missing ${EMB}" >&2
  exit 1
fi

# ---- env --------------------------------------------------------------------
# `set -u` off across this block: /etc/bashrc references BASHRCSOURCED unset, and
# conda's activate hooks trip nounset too.
set +u
source ~/.bashrc
conda activate torchenv
set -u
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export TORCH_BLAS_PREFER_HIPBLASLT=0   # AMD carry-over; no-op here
# 8 threads to match the published run (folds_engaging.slurm --cpus-per-task=8), so
# BLAS reduction order matches and the combined_e1 identity gate can be tight.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export WANDB_MODE=disabled
export MPLCONFIGDIR=/tmp/matplotlib
mkdir -p "${OUT_DIR}"

# predict_combined.py:343 builds ModelCheckpoint with no dirpath, so probe
# checkpoints land in the CWD. The published run littered 2101 files / 2.7 GB into
# the repo root that way, and POOL is at ~93%. Run from node-local /tmp and delete.
SCRATCH="/tmp/${USER}/rebut_probe_${SLURM_ARRAY_JOB_ID:-0}_${IDX}"
mkdir -p "${SCRATCH}"
cd "${SCRATCH}" || exit 1

cleanup() {
  rc=$?
  cd /tmp || true
  rm -rf "${SCRATCH}"
  if [ "${rc}" -ne 0 ]; then
    _ping "**[rebuttal concat] FAILED** \`${VARIANT}\` fold ${FOLD} (rc=${rc}, job ${SLURM_ARRAY_JOB_ID:-?}_${IDX}). See \`rebuttal_experiments/logs/\`."
  fi
  exit "${rc}"
}
trap cleanup EXIT

python "${PREDICT}" \
  --embeddings "${EMB}" \
  --out        "${OUT}" \
  --device cpu \
  --mlp-arch aion \
  --seed 0 \
  --n-folds "${N_FOLDS}" \
  --fold "${FOLD}"

echo "[done] ${OUT}"
wc -l "${OUT}"
