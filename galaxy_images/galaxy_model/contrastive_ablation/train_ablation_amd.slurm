#!/bin/bash
#SBATCH --job-name=contr-abl
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --partition=mi2101x
#SBATCH --output=/work1/jeroenaudenaert/pablomer/logs/%x_%j.out
#SBATCH --error=/work1/jeroenaudenaert/pablomer/logs/%x_%j.err
#SBATCH --open-mode=append
#
# Contrastive ablation training — AMD cluster (MI210). TRAINING ONLY.
# The downstream evaluation runs on Engaging (it needs the downstream image
# stores under /orcd/.../scratch, which do not exist here); see RUN_INSTRUCTIONS.md.
#
# On AMD the GPU comes with the partition, so there is no --gres line.
# mi2101x = 1x MI210, 12h wall. This job SELF-CHAINS: it queues an afterany
# successor before training starts, so a wall-limit kill resumes from last.ckpt.
# The python entry writes a DONE marker at --max-steps; the next link sees it and exits.
#
# Defaults reproduce the published contrastive-spatial-conv1x1 arm plus the DWNVAL
# holdout. Every other arm is one env override; see RUN_INSTRUCTIONS.md for the queue.
#
#   # A. holdout hygiene (default)
#   sbatch train_ablation_amd.slurm
#
#   # B. instrument negatives restricted to the anchor's own survey
#   sbatch --export=ALL,RUN_TAG=instrneg-samesurvey,INSTRUMENT_NEGATIVES=same_survey,HOLDOUT=0 \
#          train_ablation_amd.slurm
#
#   # C. galaxy negatives restricted to the positive's survey
#   sbatch --export=ALL,RUN_TAG=galneg-samesurvey,GALAXY_NEGATIVES=same_survey,HOLDOUT=0 \
#          train_ablation_amd.slurm
#
#   # D. no SimCLR head — InfoNCE on the representation the probes read
#   sbatch --export=ALL,RUN_TAG=nohead,PROJECTION_HEAD=0,HOLDOUT=0 train_ablation_amd.slurm
#
#   sbatch --partition=mi2104x --time=24:00:00 ...        # if mi2101x is congested
#
# Optional env: DISCORD_WEBHOOK (progress pings), WANDB_API_KEY (online logging).

set -eo pipefail

REPO_ROOT="${REPO_ROOT:-/work1/jeroenaudenaert/pablomer/tess-generative}"
CA_DIR="${REPO_ROOT}/galaxy_images/galaxy_model/contrastive_ablation"

RUN_TAG="${RUN_TAG:-spatial-conv1x1-DWNVAL}"
ENCODER_POOL="${ENCODER_POOL:-conv1x1}"
RANDOM_NEIGHBORS="${RANDOM_NEIGHBORS:-0}"
MAX_STEPS="${MAX_STEPS:-100000}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-15}"
# Ablation axes. Defaults = published behaviour, so each override isolates one variable.
GALAXY_NEGATIVES="${GALAXY_NEGATIVES:-mixed}"
INSTRUMENT_NEGATIVES="${INSTRUMENT_NEGATIVES:-mixed}"
PROJECTION_HEAD="${PROJECTION_HEAD:-1}"
# Raise to 128 for the negative-COUNT control: restricting negatives to one survey also
# halves how many there are, and InfoNCE quality depends on the count.
BATCH_SIZE="${BATCH_SIZE:-64}"
# HOLDOUT=1 (default) excludes the downstream-eval galaxies; 0 trains on all anchors,
# which is what the mechanism arms (B-D) want so they compare to the published run.
HOLDOUT="${HOLDOUT:-1}"

# AMD data paths. The 160x160 mmap store is the one the *-DWNVAL flow-matching runs
# used here, so `efficient` keeps this arm consistent with how DWNVAL is already done
# on this cluster. (`ram48` needs a store built by contrastive_baseline/build_ram48.py
# plus a --mem override; see RUN_INSTRUCTIONS.md.)
LOADER="${LOADER:-efficient}"
DATA_DIR="${DATA_DIR:-/work1/jeroenaudenaert/pablomer/data/neighbors_efficient}"
# Prefer the tracked copy in the repo so both clusters provably use the same list;
# AMD also has /work1/jeroenaudenaert/pablomer/data/downstream_ids/holdout_legacy_ids.txt
# (identical content) if you'd rather point at the pre-existing file.
HOLDOUT_TXT="${HOLDOUT_TXT:-${CA_DIR}/holdout_legacy_ids.txt}"

CKPT_DIR="${CKPT_DIR:-/work1/jeroenaudenaert/pablomer/outputs/contrastive_ablation/contrastive-${RUN_TAG}}"
mkdir -p "${CKPT_DIR}" /work1/jeroenaudenaert/pablomer/logs

_ping() {
    [[ -z "${DISCORD_WEBHOOK:-}" ]] && return 0
    curl -sS -X POST -H 'Content-Type: application/json' \
        -d "{\"content\": \"$1\"}" "${DISCORD_WEBHOOK}" >/dev/null 2>&1 || true
}

# --- Stop the chain if training already finished ---
if [[ -f "${CKPT_DIR}/DONE" ]]; then
    echo "[chain] ${CKPT_DIR}/DONE exists — training complete, nothing to do."
    exit 0
fi

# --- Bound crash loops ---
ATTEMPT=$(cat "${CKPT_DIR}/attempts" 2>/dev/null || echo 0)
ATTEMPT=$((ATTEMPT + 1))
echo "${ATTEMPT}" > "${CKPT_DIR}/attempts"
echo "[chain] attempt ${ATTEMPT}/${MAX_ATTEMPTS} for contrastive-${RUN_TAG} (job ${SLURM_JOB_ID:-?})"
if [[ "${ATTEMPT}" -gt "${MAX_ATTEMPTS}" ]]; then
    _ping "contr-${RUN_TAG} ❌ exceeded ${MAX_ATTEMPTS} attempts — giving up (job ${SLURM_JOB_ID:-?})"
    exit 1
fi

# --- Queue the successor now, so a wall-limit kill still continues the run ---
sbatch --dependency=afterany:${SLURM_JOB_ID} \
       --partition="${SLURM_JOB_PARTITION:-mi2101x}" \
       --export=ALL,RUN_TAG=${RUN_TAG},ENCODER_POOL=${ENCODER_POOL},RANDOM_NEIGHBORS=${RANDOM_NEIGHBORS},MAX_STEPS=${MAX_STEPS},MAX_ATTEMPTS=${MAX_ATTEMPTS},LOADER=${LOADER},DATA_DIR=${DATA_DIR},HOLDOUT_TXT=${HOLDOUT_TXT},HOLDOUT=${HOLDOUT},GALAXY_NEGATIVES=${GALAXY_NEGATIVES},INSTRUMENT_NEGATIVES=${INSTRUMENT_NEGATIVES},PROJECTION_HEAD=${PROJECTION_HEAD},BATCH_SIZE=${BATCH_SIZE},CKPT_DIR=${CKPT_DIR},REPO_ROOT=${REPO_ROOT} \
       "${CA_DIR}/train_ablation_amd.slurm" || echo "[chain] WARNING: could not queue successor"

source ~/.bashrc
conda activate torchenv
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

# ROCm: hipBLASLt is buggy on MI210 for some matrix shapes. Both halves of the
# workaround are required — this env var, and the guarded
# torch.backends.cuda.preferred_blas_library("hipblas") call in the python entry.
export TORCH_BLAS_PREFER_HIPBLASLT=0

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MPLCONFIGDIR=/tmp/matplotlib
export NUMBA_CACHE_DIR=/tmp/numba_cache
mkdir -p /tmp/matplotlib /tmp/numba_cache

# WandB online only if credentials exist; never fail the run over logging.
if [[ -n "${WANDB_API_KEY:-}" ]] || grep -q "api.wandb.ai" ~/.netrc 2>/dev/null; then
    export WANDB_MODE=online
else
    export WANDB_MODE="${WANDB_MODE:-offline}"
fi
export WANDB_DIR="${CKPT_DIR}"
echo "[wandb] WANDB_MODE=${WANDB_MODE}  WANDB_DIR=${WANDB_DIR}"

cd "${REPO_ROOT}"

# --- Preflight on the first attempt: refuse to burn a GPU on a broken holdout ---
if [[ "${ATTEMPT}" -eq 1 ]]; then
    echo "[preflight] verifying holdout resolution before training"
    echo "[preflight] note: the holdout report below is informational; the actual exclusion is controlled by HOLDOUT=${HOLDOUT}"
    python "${CA_DIR}/preflight_holdout.py" \
        --loader "${LOADER}" \
        --data-dir "${DATA_DIR}" \
        --holdout-ids-txt "${HOLDOUT_TXT}" \
        --smoke \
        --benchmark "${BENCH_BATCHES:-30}" \
        --bench-workers "${SLURM_CPUS_PER_TASK:-8}" \
        --bench-max-steps "${MAX_STEPS}"
    # The benchmark only WARNS about a slow loader (it does not fail the job) — read the
    # projected wall clock in the log. If it exceeds the allocation you can afford, cancel,
    # build the ram48 store, and resubmit with LOADER=ram48.
    _ping "contr-${RUN_TAG} 🚀 training started on AMD · pool=${ENCODER_POOL} loader=${LOADER} · target ${MAX_STEPS} steps · job ${SLURM_JOB_ID:-?}"
fi

# Holdout is a flag pair, not a value: pass the id list, or explicitly acknowledge none.
if [[ "${HOLDOUT}" == "1" ]]; then
    HOLDOUT_ARGS=(--holdout-ids-txt "${HOLDOUT_TXT}")
else
    HOLDOUT_ARGS=(--no-holdout)
fi

srun python -u "${CA_DIR}/train_contrastive_ablation.py" \
    --encoder-pool          "${ENCODER_POOL}" \
    --random-neighbors      "${RANDOM_NEIGHBORS}" \
    --max-steps             "${MAX_STEPS}" \
    --loader                "${LOADER}" \
    --data-dir              "${DATA_DIR}" \
    "${HOLDOUT_ARGS[@]}" \
    --galaxy-negatives      "${GALAXY_NEGATIVES}" \
    --instrument-negatives  "${INSTRUMENT_NEGATIVES}" \
    --projection-head       "${PROJECTION_HEAD}" \
    --ckpt-dir              "${CKPT_DIR}" \
    --run-name              "contrastive-${RUN_TAG}" \
    --batch-size            "${BATCH_SIZE}" \
    --num-workers           "${SLURM_CPUS_PER_TASK:-8}"

if [[ -f "${CKPT_DIR}/DONE" ]]; then
    echo "[done] training complete. Checkpoints in ${CKPT_DIR}:"
    ls -la "${CKPT_DIR}"
    _ping "contr-${RUN_TAG} ✅ training finished (${MAX_STEPS} steps) on AMD. Copy best-*.ckpt to Engaging and run contrastive_ablation/eval_dwnval_engaging.slurm."
    echo
    echo "NEXT STEP (on Engaging, not here):"
    echo "  1. rsync the best checkpoint to /orcd/pool/007/pablomer/checkpoints_new/contrastive-${RUN_TAG}/"
    echo "  2. sbatch --export=ALL,RUN_TAG=${RUN_TAG} \\"
    echo "       galaxy_images/galaxy_model/contrastive_ablation/eval_ablation_engaging.slurm"
fi

echo "Job finished at $(date)"
