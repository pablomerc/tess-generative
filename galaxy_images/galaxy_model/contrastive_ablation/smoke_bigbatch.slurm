#!/bin/bash
#SBATCH --job-name=contr-smoke640
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:40:00
#SBATCH --partition=mi2101x
#SBATCH --mem=96G
#SBATCH --output=/work1/jeroenaudenaert/pablomer/logs/%x_%j.out
#SBATCH --error=/work1/jeroenaudenaert/pablomer/logs/%x_%j.err
#
# 30-step GPU smoke test for the large-batch contrastive arm. ~10 min.
#
# Answers the three things the CPU preflight cannot, before a multi-link allocation is
# committed: (1) does batch 640 fit in GPU memory, (2) what is the real step rate at
# batch 640 (the loader benchmark alone does not tell you this), (3) do the permanent
# step snapshots actually get written and reload.
#
# AMD (defaults):
#   sbatch galaxy_images/galaxy_model/contrastive_ablation/smoke_bigbatch.slurm
#
# Engaging (needs --gres, its own paths and log dir):
#   sbatch --partition=mit_normal_gpu --gres=gpu:1 --mem=64G --time=00:40:00 \
#     --output=/orcd/pool/007/pablomer/tess-generative/galaxy_images/galaxy_model/contrastive_ablation/logs/%x_%j.out \
#     --error=/orcd/pool/007/pablomer/tess-generative/galaxy_images/galaxy_model/contrastive_ablation/logs/%x_%j.err \
#     --export=ALL,REPO_ROOT=/orcd/pool/007/pablomer/tess-generative,\
# DATA_DIR=/orcd/pool/007/pablomer/efficient_neighs_48,\
# CKPT_DIR=/orcd/pool/007/pablomer/smoke_bs640 \
#     galaxy_images/galaxy_model/contrastive_ablation/smoke_bigbatch.slurm
#
# Env: BATCH_SIZE (640), MAX_STEPS (30), LOADER (ram48), DATA_DIR, CKPT_DIR, DISCORD_WEBHOOK.
# The checkpoint dir is a throwaway — the script deletes it on success unless KEEP=1.

set -eo pipefail

REPO_ROOT="${REPO_ROOT:-/work1/jeroenaudenaert/pablomer/tess-generative}"
CA_DIR="${REPO_ROOT}/galaxy_images/galaxy_model/contrastive_ablation"

BATCH_SIZE="${BATCH_SIZE:-640}"
MAX_STEPS="${MAX_STEPS:-30}"
LOADER="${LOADER:-ram48}"
DATA_DIR="${DATA_DIR:-/work1/jeroenaudenaert/pablomer/data/efficient_neighs_48}"
HOLDOUT_TXT="${HOLDOUT_TXT:-${CA_DIR}/holdout_legacy_ids.txt}"
CKPT_DIR="${CKPT_DIR:-/work1/jeroenaudenaert/pablomer/outputs/contrastive_ablation/smoke-bs${BATCH_SIZE}}"
SNAP_A=$((MAX_STEPS / 3))
SNAP_B=$((2 * MAX_STEPS / 3))

_ping() {
    [[ -z "${DISCORD_WEBHOOK:-}" ]] && return 0
    curl -sS -X POST -H 'Content-Type: application/json' \
        -d "{\"content\": \"$1\"}" "${DISCORD_WEBHOOK}" >/dev/null 2>&1 || true
}

# A leftover DONE from a previous smoke would make the trainer's completion check
# meaningless, so start from a clean directory every time.
rm -rf "${CKPT_DIR}"
mkdir -p "${CKPT_DIR}"

source ~/.bashrc
conda activate torchenv
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export TORCH_BLAS_PREFER_HIPBLASLT=0        # AMD fix, harmless no-op on NVIDIA
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MPLCONFIGDIR=/tmp/matplotlib
export NUMBA_CACHE_DIR=/tmp/numba_cache
mkdir -p /tmp/matplotlib /tmp/numba_cache
cd "${REPO_ROOT}"

echo "=== [1/3] loader benchmark at batch ${BATCH_SIZE} ==========================="
python "${CA_DIR}/preflight_holdout.py" \
    --loader "${LOADER}" \
    --data-dir "${DATA_DIR}" \
    --holdout-ids-txt "${HOLDOUT_TXT}" \
    --batch-size "${BATCH_SIZE}" \
    --smoke \
    --benchmark 10 \
    --bench-workers "${SLURM_CPUS_PER_TASK:-8}" \
    --bench-max-steps 100000

echo "=== [2/3] ${MAX_STEPS} real training steps at batch ${BATCH_SIZE} =========="
SECONDS=0
srun python -u "${CA_DIR}/train_contrastive_ablation.py" \
    --encoder-pool          conv1x1 \
    --random-neighbors      0 \
    --max-steps             "${MAX_STEPS}" \
    --loader                "${LOADER}" \
    --data-dir              "${DATA_DIR}" \
    --holdout-ids-txt       "${HOLDOUT_TXT}" \
    --ckpt-dir              "${CKPT_DIR}" \
    --run-name              "smoke-bs${BATCH_SIZE}" \
    --batch-size            "${BATCH_SIZE}" \
    --snapshot-steps        "${SNAP_A}:${SNAP_B}" \
    --num-umap-batches      1 \
    --val-check-interval    "${SNAP_B}" \
    --limit-val-batches     2 \
    --num-workers           "${SLURM_CPUS_PER_TASK:-8}" \
    --no-wandb
echo "[smoke] wall clock for setup + ${MAX_STEPS} steps: ${SECONDS}s"

echo "=== [3/3] verifying artifacts =============================================="
python - "${CKPT_DIR}" "${SNAP_A}" "${SNAP_B}" <<'PY'
import sys
from pathlib import Path
import torch

ckpt_dir, snap_a, snap_b = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
failures = []

if not (ckpt_dir / "DONE").exists():
    failures.append("no DONE marker — the run did not reach --max-steps")
else:
    print("[check] DONE:\n  " + (ckpt_dir / "DONE").read_text().strip().replace("\n", "\n  "))

for step in (snap_a, snap_b):
    p = ckpt_dir / f"snapshot-step={step}.ckpt"
    if not p.exists():
        failures.append(f"missing {p.name}")
        continue
    got = int(torch.load(p, map_location="cpu", weights_only=False)["global_step"])
    if got != step:
        failures.append(f"{p.name} records global_step={got}, expected {step}")
    else:
        print(f"[check] {p.name}: global_step={got}, {p.stat().st_size / 2**20:.0f} MiB  OK")

if failures:
    print("SMOKE FAILED:")
    for f in failures:
        print(f"  - {f}")
    raise SystemExit(1)
print("SMOKE OK")
PY

_ping "contr-smoke batch=${BATCH_SIZE} ✅ ${MAX_STEPS} steps + snapshots OK (job ${SLURM_JOB_ID:-?}) — see log for step rate and peak GPU memory"

if [[ "${KEEP:-0}" != "1" ]]; then
    rm -rf "${CKPT_DIR}"
    echo "[smoke] removed ${CKPT_DIR} (KEEP=1 to keep it)"
fi
echo "Done at $(date)"
