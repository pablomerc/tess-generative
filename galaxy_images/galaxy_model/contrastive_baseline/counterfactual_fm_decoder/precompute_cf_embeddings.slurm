#!/bin/bash
#SBATCH --job-name=cf-precompute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --partition=pg_mki_aryeh
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --open-mode=append
#
# Stage 1: encode all anchors' samegal + neighbors through the frozen contrastive
# dual-encoder and write the CF [embedding, image] H5.
#
# Reads the ram48 store (efficient_neighs_48). The previous attempt read the
# 160x160 efficient_neighs mmap store and managed ~1 anchor/s => ~29 h, which is
# what killed job 18875467 at its wall limit. ram48 is a sequential 16 GB slurp
# into RAM, so the pass is GPU-bound and finishes in minutes; it also matches the
# encoder's own training inputs and the eval loader's targets exactly.
# mem=96G covers the two 8.6 GB float16 bins + catalog + workers.
#
# Submit:
#   sbatch precompute_cf_embeddings.slurm
# Then:
#   sbatch --dependency=afterok:<PRE_JID> train_cf_fm_tokens.slurm

set -eo pipefail

# ---- EDIT THESE FOR YOUR CLUSTER ----
REPO=/orcd/pool/007/pablomer/tess-generative
TORCHENV_PY=/orcd/home/002/pablomer/miniforge3/envs/torchenv/bin/python
DATA_DIR=/orcd/pool/007/pablomer/efficient_neighs_48
LOADER=ram48
CKPT=/orcd/pool/007/pablomer/outputs/contrastive/best-epoch=599-step=240000.ckpt
OUT_H5=/orcd/pool/007/pablomer/outputs/counterfactual_fm/cf_embeddings_all.h5
BATCH=128
MAX_NEIGHBORS=5
# Milestone snapshots were insurance against the 29 h efficient-store pass. The ram48
# pass is short enough that they'd cost ~1.3 GB on a 90%-full pool for nothing.
SNAPSHOT_AT=
WEBHOOK="${DISCORD_WEBHOOK:-https://discord.com/api/webhooks/1500242707522519183/NqqAxuo5n0Ex4X81vMcq7gDOjrP3oAURAVZtgrZT5CramGNfBGPxuEvC2NnxwYBkfs9j}"
# -------------------------------------

_ping() { curl -sS -X POST -H 'Content-Type: application/json' -d "{\"content\": \"$1\"}" "${WEBHOOK}" >/dev/null 2>&1 || true; }
_ping "cf-precompute 🚀 start · job ${SLURM_JOB_ID:-?} · host ${SLURMD_NODENAME:-?} · loader=${LOADER} · 3h wall"
cleanup() {
  local rc=$?
  if [[ ${rc} -eq 0 ]]; then
    _ping "cf-precompute ✅ done · \`${OUT_H5}\` · job ${SLURM_JOB_ID:-?}"
  else
    _ping "cf-precompute ❌ failed (exit ${rc}) · job ${SLURM_JOB_ID:-?} · no H5 promoted (atomic .tmp)"
  fi
}
trap cleanup EXIT

source ~/.bashrc
conda activate torchenv || true

export PYTHONPATH="$REPO:$PYTHONPATH"
cd "$REPO"

# Drop the previous corrupt/partial final if present (script also .bak's it).
if [[ -f "$OUT_H5" ]]; then
  echo "[precompute] removing previous $OUT_H5"
  rm -f "$OUT_H5"
fi
rm -f "${OUT_H5}.tmp"

echo "PYTHON=$TORCHENV_PY"
"$TORCHENV_PY" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

srun "$TORCHENV_PY" -u galaxy_images/galaxy_model/contrastive_baseline/counterfactual_fm_decoder/precompute_cf_embeddings.py \
    --data-dir "$DATA_DIR" \
    --loader "$LOADER" \
    --ckpt "$CKPT" \
    --out-h5 "$OUT_H5" \
    --batch-size "$BATCH" \
    --max-neighbors "$MAX_NEIGHBORS" \
    --target-dtype float16 \
    --snapshot-at "$SNAPSHOT_AT" \
    --discord-webhook "$WEBHOOK"

echo "CF precompute done at $(date)"
ls -lh "$(dirname "$OUT_H5")"/cf_embeddings_*.h5 || true
