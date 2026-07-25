#!/bin/bash
#SBATCH --job-name=shuffle-ablation
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --partition=mit_normal_gpu
#SBATCH --mem=64G
#SBATCH --chdir=/home/pablomer/orcd/pool/tess-generative
#SBATCH --output=/home/pablomer/orcd/pool/tess-generative/galaxy_images/galaxy_model/shuffle_ablation/logs/%x_%j.out
#SBATCH --error=/home/pablomer/orcd/pool/tess-generative/galaxy_images/galaxy_model/shuffle_ablation/logs/%x_%j.err
#SBATCH --open-mode=append

# Shuffle-conditioning ablation (reviewer SYJm Q2).
# Defaults: paper base checkpoint, M=8, 250 Euler steps, C0–C3.
# Override via env: N_SAMPLES=1 NUM_STEPS=25 for a smoke run.

set -eo pipefail

SCRIPT_DIR="/home/pablomer/orcd/pool/tess-generative/galaxy_images/galaxy_model/shuffle_ablation"
REPO_ROOT="/home/pablomer/orcd/pool/tess-generative"

DISCORD_WEBHOOK="${DISCORD_WEBHOOK:-}"
_ping() {
    if [[ -z "${DISCORD_WEBHOOK}" ]]; then
        echo "[ping] $1"
        return 0
    fi
    curl -sS -X POST -H 'Content-Type: application/json' \
        -d "{\"content\": $(python -c 'import json,sys; print(json.dumps(sys.argv[1]))' "$1")}" \
        "${DISCORD_WEBHOOK}" >/dev/null || true
}
_ping_file() {
    if [[ -z "${DISCORD_WEBHOOK}" ]]; then
        echo "[ping] $1 (file: $2)"
        return 0
    fi
    if [[ -f "$2" ]]; then
        curl -sS -F "payload_json={\"content\":$(python -c 'import json,sys; print(json.dumps(sys.argv[1]))' "$1")}" \
            -F "file=@$2" "${DISCORD_WEBHOOK}" >/dev/null || true
    else
        _ping "$1 (file missing: $2)"
    fi
}
_on_exit() {
    rc=$?
    if [[ $rc -ne 0 ]]; then
        _ping "shuffle-ablation ${SLURM_JOB_ID:-?} ❌ failed (rc=$rc) at $(date '+%F %T')"
    fi
}
trap _on_exit EXIT

source ~/.bashrc
conda activate torchenv
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export TORCH_BLAS_PREFER_HIPBLASLT=0
export OMP_NUM_THREADS=1
export WANDB_MODE=disabled

cd "${REPO_ROOT}"
mkdir -p "${SCRIPT_DIR}/logs" "${SCRIPT_DIR}/results"

N_SAMPLES="${N_SAMPLES:-8}"
M_LIST="${M_LIST:-}"          # e.g. "0,1,2,3" to split M across jobs (overrides N_SAMPLES)
NUM_STEPS="${NUM_STEPS:-250}"
N="${N:-256}"
M_TABLE="${M_TABLE:-}"        # which m the tables/figures summarise (default: lowest in this run)
OUT_DIR="${OUT_DIR:-${SCRIPT_DIR}/results}"
CKPT="${CKPT:-${REPO_ROOT}/galaxy_images/galaxy_model/checkpoints/base/snapshot.ckpt}"
CONFIG="${CONFIG:-${REPO_ROOT}/galaxy_images/galaxy_model/diffusion_ablation/configs/neighbors_fm_control.json}"

_ping "shuffle-ablation ${SLURM_JOB_ID:-?} 🚀 started — n=${N} M=${M_LIST:-${N_SAMPLES}} steps=${NUM_STEPS}"

python -m galaxy_images.galaxy_model.shuffle_ablation.run_shuffle_ablation \
    --config "${CONFIG}" \
    --checkpoint "${CKPT}" \
    --n "${N}" \
    --n-samples "${N_SAMPLES}" \
    --m-list "${M_LIST}" \
    --num-steps "${NUM_STEPS}" \
    --out-dir "${OUT_DIR}" \
    --discord-webhook "${DISCORD_WEBHOOK}" \
    --ping-prefix "shuffle-ablation ${SLURM_JOB_ID:-?}"

# Tables/figures summarise one m; default to the lowest index present in this run.
if [[ -z "${M_TABLE}" ]]; then
    M_TABLE=$(python -c 'import sys,csv; print(min(int(r["m"]) for r in csv.DictReader(open(sys.argv[1]))))' "${OUT_DIR}/shuffle_per_anchor.csv")
fi

python -m galaxy_images.galaxy_model.shuffle_ablation.make_tables \
    --csv "${OUT_DIR}/shuffle_per_anchor.csv" \
    --out-dir "${OUT_DIR}/tables" \
    --m "${M_TABLE}"

python -m galaxy_images.galaxy_model.shuffle_ablation.make_figures \
    --results-dir "${OUT_DIR}" \
    --out-dir "${OUT_DIR}/figures" \
    --m "${M_TABLE}"

_ping_file "shuffle-ablation ${SLURM_JOB_ID:-?} ✅ finished" "${OUT_DIR}/tables/summary_mean_sem.md"
if [[ -f "${OUT_DIR}/figures/qualitative_grid.png" ]]; then
    _ping_file "shuffle-ablation qualitative grid" "${OUT_DIR}/figures/qualitative_grid.png"
fi

echo "Done at $(date)"
