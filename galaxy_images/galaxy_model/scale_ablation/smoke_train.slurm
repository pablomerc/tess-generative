#!/bin/bash
#SBATCH --job-name=scale-smoke
#SBATCH --partition=mit_normal_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/home/pablomer/orcd/pool/tess-generative/galaxy_images/galaxy_model/scale_ablation/logs/%x_%j.out
#SBATCH --error=/home/pablomer/orcd/pool/tess-generative/galaxy_images/galaxy_model/scale_ablation/logs/%x_%j.err
#SBATCH --open-mode=append
#
# 200-step GPU smoke for one rung. Parameterize with:
#   sbatch --export=ALL,CONFIG_PATH=.../scale_ablation/configs/scale_1000.json \
#          galaxy_images/galaxy_model/scale_ablation/smoke_train.slurm
#
# Asserts the three things unique to this ablation, straight out of the log:
#   1. the downstream holdout actually fires in ram48 mode (it was a silent no-op before)
#   2. the split lands at kept=98,275 / train=93,362 / val=4,913
#   3. the rung is tiled to 93,362 items, so steps/epoch (and hence the epoch-driven LR
#      waveform) is identical to every other arm

set -eo pipefail

REPO="/home/pablomer/orcd/pool/tess-generative"
ABL_DIR="${REPO}/galaxy_images/galaxy_model/scale_ablation"
mkdir -p "${ABL_DIR}/logs"

: "${CONFIG_PATH:?set CONFIG_PATH}"
CONFIG_BASENAME="$(basename "${CONFIG_PATH}")"

WEBHOOK="${DISCORD_WEBHOOK:-https://discord.com/api/webhooks/1500242707522519183/NqqAxuo5n0Ex4X81vMcq7gDOjrP3oAURAVZtgrZT5CramGNfBGPxuEvC2NnxwYBkfs9j}"
_ping() { curl -sS -X POST -H 'Content-Type: application/json' -d "{\"content\": \"$1\"}" "${WEBHOOK}" >/dev/null 2>&1 || true; }

_ping "scale-smoke 🚀 start · \`${CONFIG_BASENAME}\` · job ${SLURM_JOB_ID:-?}"
trap '_ping "scale-smoke $([ $? -eq 0 ] && echo ✅ || echo ❌) · \`${CONFIG_BASENAME}\` · job ${SLURM_JOB_ID:-?}"' EXIT

source ~/.bashrc
conda activate torchenv
cd "${REPO}"
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export TORCH_BLAS_PREFER_HIPBLASLT=0
# Offline (not disabled): the module's validation calls the W&B-specific
# self.logger.experiment.log(), which crashes under the CSV fallback logger.
export WANDB_MODE=offline

export SMOKE_OUT="/orcd/pool/007/pablomer/checkpoints_new/scale_ablation/smoke/${CONFIG_BASENAME%.json}"
mkdir -p "${SMOKE_OUT}"

# --unbuffered: srun otherwise block-buffers stdout and a slow RAM48 load looks
# identical to a hang (job 18790012 showed 0 bytes of output for 23 minutes).
srun --unbuffered python -u -m galaxy_images.galaxy_model.train \
  --config "${CONFIG_PATH}" \
  --set trainer.devices=1 \
  --set trainer.num_steps=200 \
  --set trainer.val_check_interval=100 \
  --set trainer.checkpoint_every_n_train_steps=100 \
  --set data.batch_size=16 \
  --set "wandb.name=smoke-${CONFIG_BASENAME%.json}" \
  --set trainer.monitor_metric=val/loss \
  --set model.num_integration_steps=10 \
  --set model.num_mse_images=8 \
  --set model.num_sample_images=2 \
  --set "run.output_dir=${SMOKE_OUT}" \
  --set run.shared_checkpoint_dir=null

# ---- post-smoke assertions -------------------------------------------------------
LOG="${ABL_DIR}/logs/${SLURM_JOB_NAME:-scale-smoke}_${SLURM_JOB_ID}.out"
echo "[smoke-check] scanning ${LOG}"
fail=0
_want() {  # _want <description> <grep-pattern>
  if grep -qF "$2" "${LOG}"; then
    echo "[smoke-check] ok   : $1"
  else
    echo "[smoke-check] FAIL : $1 (expected to find: $2)" >&2
    fail=1
  fi
}
_want "holdout fired in ram48 mode"  "matched anchors: 5,466 / 103,741"
_want "split is kept/train/val"      "kept=98,275 | val_ratio=0.05 -> train=93,362, val=4,913"
_want "rung tiled to full length"    "tiled to 93,362 items"
_want "dataset loaded from RAM48"    "103,741 anchors / 468,197 rows"

export SMOKE_FAIL="${fail}"
python -u - <<'PY'
import os, sys
from pathlib import Path

smoke_out = Path(os.environ["SMOKE_OUT"])
# periodic ckpt keeps auto_insert_metric_name=True, so the real name has a doubled 'step='
ckpts = sorted(smoke_out.rglob("latest-step*.ckpt"))
print(f"[smoke-check] found ckpts: {[p.name for p in ckpts]}")
if not ckpts:
    print("[smoke-check] ERROR: no latest-step*.ckpt found", file=sys.stderr)
    sys.exit(1)

from galaxy_images.galaxy_model.double_train_fm_neighbors import ConditionalFlowMatchingModule
model = ConditionalFlowMatchingModule.load_from_checkpoint(str(ckpts[-1]), map_location="cpu")
n_par = sum(p.numel() for p in model.parameters())
print(f"[smoke-check] FM reload ok: {type(model).__name__}, {n_par/1e6:.1f}M params")

if os.environ.get("SMOKE_FAIL") != "0":
    print("[smoke-check] FAIL: one or more log assertions failed", file=sys.stderr)
    sys.exit(1)
print("[smoke-check] PASS")
PY
