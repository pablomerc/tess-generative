#!/bin/bash
#SBATCH --job-name=contr-abl-eng
#SBATCH --partition=pg_mki_aryeh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=48:00:00
#SBATCH --output=/orcd/pool/007/pablomer/tess-generative/galaxy_images/galaxy_model/contrastive_ablation/logs/%x_%j.out
#SBATCH --error=/orcd/pool/007/pablomer/tess-generative/galaxy_images/galaxy_model/contrastive_ablation/logs/%x_%j.err
#SBATCH --open-mode=append
#
# Contrastive ablation training — ENGAGING (pg_mki_aryeh, 1x H200). TRAINING ONLY.
#
# Engaging twin of train_ablation_amd.slurm. Same env knobs, same python entry, same
# defaults for every scientific axis; only the cluster-specific parts differ (per
# CLAUDE.md: keep both variants rather than merging them):
#
#   * #SBATCH --gres is REQUIRED here (AMD gets its GPU from the partition), and the
#     self-chained successor must be given it explicitly too — a chain that forgets
#     --gres resumes on a CPU-only node and dies. node[4701-4702] in this partition have
#     no GPUs, so --gres is also what pins the job to node4900.
#   * paths are Engaging's, and the run dir lives in the 200TB mki_aryeh group
#     allocation (POOL was at 92% on 2026-07-28), symlinked into checkpoints_new/ so the
#     eval script finds it at its default location.
#   * TORCH_BLAS_PREFER_HIPBLASLT is deliberately NOT set: it is what triggers the
#     python entry's preferred_blas_library("hipblas") call, which has no business
#     running on CUDA.
#   * this partition allows 48h, so a 100k-step run at batch 640 should finish in one
#     link. The chain stays as insurance.
#
# Defaults reproduce the published contrastive-spatial-conv1x1 arm plus the DWNVAL
# holdout, at batch 64. Arm I (batch 640) is one override set — see RUN_BIGBATCH.md:
#
#   sbatch --export=ALL,RUN_TAG=spatial-conv1x1-BS640-DWNVAL,BATCH_SIZE=640,\
# SNAPSHOT_STEPS=10000:35000,NUM_UMAP_BATCHES=1 \
#     galaxy_images/galaxy_model/contrastive_ablation/train_ablation_engaging.slurm
#
# Optional env: DISCORD_WEBHOOK, WANDB_API_KEY, MAX_STEPS, LR, HOLDOUT, LOADER, DATA_DIR,
#               CKPT_DIR, MAX_ATTEMPTS, CHAIN_{PARTITION,TIME,MEM,CPUS,GRES}.

set -eo pipefail

REPO_ROOT="${REPO_ROOT:-/orcd/pool/007/pablomer/tess-generative}"
CA_DIR="${REPO_ROOT}/galaxy_images/galaxy_model/contrastive_ablation"

RUN_TAG="${RUN_TAG:-spatial-conv1x1-DWNVAL}"
ENCODER_POOL="${ENCODER_POOL:-conv1x1}"
RANDOM_NEIGHBORS="${RANDOM_NEIGHBORS:-0}"
MAX_STEPS="${MAX_STEPS:-100000}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-10}"
GALAXY_NEGATIVES="${GALAXY_NEGATIVES:-mixed}"
INSTRUMENT_NEGATIVES="${INSTRUMENT_NEGATIVES:-mixed}"
PROJECTION_HEAD="${PROJECTION_HEAD:-1}"
BATCH_SIZE="${BATCH_SIZE:-64}"
LR="${LR:-1e-4}"
# Separate multiple steps with COLONS (10000:35000) — sbatch --export splits on commas.
SNAPSHOT_STEPS="${SNAPSHOT_STEPS:-}"
NUM_UMAP_BATCHES="${NUM_UMAP_BATCHES:-8}"
HOLDOUT="${HOLDOUT:-1}"

# The prebaked 48x48 store the published Engaging contrastive arms trained from, and what
# arm A used on AMD (its DONE file records loader=ram48) — so this is the loader the A/B
# needs. It is read into RAM, hence the 96G request.
LOADER="${LOADER:-ram48}"
DATA_DIR="${DATA_DIR:-/orcd/pool/007/pablomer/efficient_neighs_48}"
HOLDOUT_TXT="${HOLDOUT_TXT:-${CA_DIR}/holdout_legacy_ids.txt}"

CKPT_DIR="${CKPT_DIR:-/orcd/data/mki_aryeh/001/pablomer/contrastive_ablation/contrastive-${RUN_TAG}}"
CKPT_LINK="${CKPT_LINK:-/orcd/pool/007/pablomer/checkpoints_new/contrastive-${RUN_TAG}}"
mkdir -p "${CKPT_DIR}" "${CA_DIR}/logs"
# Canonical path for the eval script, without spending POOL quota on the checkpoints.
if [[ ! -e "${CKPT_LINK}" ]]; then
    ln -s "${CKPT_DIR}" "${CKPT_LINK}" && echo "[ckpt] linked ${CKPT_LINK} -> ${CKPT_DIR}"
fi

WEBHOOK="${DISCORD_WEBHOOK:-https://discord.com/api/webhooks/1500242707522519183/NqqAxuo5n0Ex4X81vMcq7gDOjrP3oAURAVZtgrZT5CramGNfBGPxuEvC2NnxwYBkfs9j}"
_ping() {
    [[ -z "${WEBHOOK}" ]] && return 0
    curl -sS -X POST -H 'Content-Type: application/json' \
        -d "{\"content\": \"$1\"}" "${WEBHOOK}" >/dev/null 2>&1 || true
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
echo "[chain] attempt ${ATTEMPT}/${MAX_ATTEMPTS} for contrastive-${RUN_TAG} (job ${SLURM_JOB_ID:-?} on ${SLURMD_NODENAME:-?})"
if [[ "${ATTEMPT}" -gt "${MAX_ATTEMPTS}" ]]; then
    _ping "contr-${RUN_TAG} ❌ exceeded ${MAX_ATTEMPTS} attempts — giving up (job ${SLURM_JOB_ID:-?})"
    exit 1
fi

# --- Queue the successor now, so a wall-limit kill still continues the run ---
# sbatch resource flags are NOT inherited by a new submission, only the environment is.
# --gres is the one that matters most here: without it the successor lands on a CPU node.
CHAIN_PARTITION="${CHAIN_PARTITION:-${SLURM_JOB_PARTITION:-pg_mki_aryeh}}"
CHAIN_GRES="${CHAIN_GRES:-gpu:h200:1}"
CHAIN_TIME="${CHAIN_TIME:-48:00:00}"
CHAIN_MEM="${CHAIN_MEM:-${SLURM_MEM_PER_NODE:+${SLURM_MEM_PER_NODE}M}}"
CHAIN_CPUS="${CHAIN_CPUS:-${SLURM_CPUS_PER_TASK:-16}}"
echo "[chain] successor: partition=${CHAIN_PARTITION} gres=${CHAIN_GRES} time=${CHAIN_TIME}" \
     "mem=${CHAIN_MEM:-<script default>} cpus=${CHAIN_CPUS}"

sbatch --dependency=afterany:${SLURM_JOB_ID} \
       --partition="${CHAIN_PARTITION}" \
       --gres="${CHAIN_GRES}" \
       --time="${CHAIN_TIME}" \
       ${CHAIN_MEM:+--mem="${CHAIN_MEM}"} \
       --cpus-per-task="${CHAIN_CPUS}" \
       --export=ALL,RUN_TAG=${RUN_TAG},ENCODER_POOL=${ENCODER_POOL},RANDOM_NEIGHBORS=${RANDOM_NEIGHBORS},MAX_STEPS=${MAX_STEPS},MAX_ATTEMPTS=${MAX_ATTEMPTS},LOADER=${LOADER},DATA_DIR=${DATA_DIR},HOLDOUT_TXT=${HOLDOUT_TXT},HOLDOUT=${HOLDOUT},GALAXY_NEGATIVES=${GALAXY_NEGATIVES},INSTRUMENT_NEGATIVES=${INSTRUMENT_NEGATIVES},PROJECTION_HEAD=${PROJECTION_HEAD},BATCH_SIZE=${BATCH_SIZE},LR=${LR},SNAPSHOT_STEPS=${SNAPSHOT_STEPS},NUM_UMAP_BATCHES=${NUM_UMAP_BATCHES},CKPT_DIR=${CKPT_DIR},CKPT_LINK=${CKPT_LINK},REPO_ROOT=${REPO_ROOT},CHAIN_PARTITION=${CHAIN_PARTITION},CHAIN_GRES=${CHAIN_GRES},CHAIN_TIME=${CHAIN_TIME},CHAIN_MEM=${CHAIN_MEM},CHAIN_CPUS=${CHAIN_CPUS} \
       "${CA_DIR}/train_ablation_engaging.slurm" || echo "[chain] WARNING: could not queue successor"

source ~/.bashrc
conda activate torchenv
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MPLCONFIGDIR=/tmp/matplotlib
export NUMBA_CACHE_DIR=/tmp/numba_cache
mkdir -p /tmp/matplotlib /tmp/numba_cache

if [[ -n "${WANDB_API_KEY:-}" ]] || grep -q "api.wandb.ai" ~/.netrc 2>/dev/null; then
    export WANDB_MODE=online
else
    export WANDB_MODE="${WANDB_MODE:-offline}"
fi
export WANDB_DIR="${CKPT_DIR}"
echo "[wandb] WANDB_MODE=${WANDB_MODE}  WANDB_DIR=${WANDB_DIR}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

cd "${REPO_ROOT}"

# --- Preflight on the first attempt: refuse to burn a GPU on a broken holdout ---
if [[ "${ATTEMPT}" -eq 1 ]]; then
    echo "[preflight] verifying holdout resolution + loader throughput at batch ${BATCH_SIZE}"
    python "${CA_DIR}/preflight_holdout.py" \
        --loader "${LOADER}" \
        --data-dir "${DATA_DIR}" \
        --holdout-ids-txt "${HOLDOUT_TXT}" \
        --batch-size "${BATCH_SIZE}" \
        --smoke \
        --benchmark "${BENCH_BATCHES:-20}" \
        --bench-workers "${SLURM_CPUS_PER_TASK:-16}" \
        --bench-max-steps "${MAX_STEPS}"
    _ping "contr-${RUN_TAG} 🚀 training started on Engaging ${SLURMD_NODENAME:-?} (1x H200) · pool=${ENCODER_POOL} loader=${LOADER} batch=${BATCH_SIZE} lr=${LR}${SNAPSHOT_STEPS:+ snapshots=${SNAPSHOT_STEPS}} · target ${MAX_STEPS} steps · job ${SLURM_JOB_ID:-?}"
fi

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
    --lr                    "${LR}" \
    --snapshot-steps        "${SNAPSHOT_STEPS}" \
    --num-umap-batches      "${NUM_UMAP_BATCHES}" \
    --num-workers           "${SLURM_CPUS_PER_TASK:-16}"

if [[ -f "${CKPT_DIR}/DONE" ]]; then
    echo "[done] training complete. Checkpoints in ${CKPT_DIR}:"
    ls -la "${CKPT_DIR}"
    _ping "contr-${RUN_TAG} ✅ training finished (${MAX_STEPS} steps, batch ${BATCH_SIZE}) on Engaging. Next: sbatch --export=ALL,RUN_TAG=${RUN_TAG},REF_VAR=contrastive-spatial-conv1x1-DWNVAL contrastive_ablation/eval_ablation_engaging.slurm"
    echo
    echo "NEXT STEP (same cluster — no checkpoint copy needed):"
    echo "  sbatch --export=ALL,RUN_TAG=${RUN_TAG},REF_VAR=contrastive-spatial-conv1x1-DWNVAL \\"
    echo "    galaxy_images/galaxy_model/contrastive_ablation/eval_ablation_engaging.slurm"
    if [[ -n "${SNAPSHOT_STEPS}" ]]; then
        echo "  ...and once per snapshot, e.g. SNAP_STEP=10000 with RUN_TAG=${RUN_TAG%-DWNVAL}-eq6M-DWNVAL"
    fi
fi

echo "Job finished at $(date)"
