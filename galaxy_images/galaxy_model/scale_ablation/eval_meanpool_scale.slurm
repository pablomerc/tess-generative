#!/bin/bash
#SBATCH --job-name=meanpool-scale
#SBATCH --partition=mit_normal_gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/home/pablomer/orcd/pool/tess-generative/galaxy_images/galaxy_model/scale_ablation/logs/%x_%j.out
#SBATCH --error=/home/pablomer/orcd/pool/tess-generative/galaxy_images/galaxy_model/scale_ablation/logs/%x_%j.err
#SBATCH --chdir=/orcd/pool/007/pablomer/tess-generative
#SBATCH --open-mode=append
#
# Build the token-mean-pooled twin for every scale arm + the fm-control reference, then
# probe it. This supplies the INSTRUMENT column on the paper's own footing: the published
# "Ours (Instrument)" bar is `base-meanpool` combined_e2 (pooled 32-d), not the flat 128-d.
# Without this the arms and the reference would be compared on different latents.
#
# `base-meanpool` and `untrained-resnet-matched-meanpool` already exist and are reused.
# Idempotent: skips any twin/CSV already on disk.

set -eo pipefail

E="/home/pablomer/orcd/pool/tess-generative/galaxy_images/galaxy_model/downstream_evaluation/engaging"
ABL="/home/pablomer/orcd/pool/tess-generative/galaxy_images/galaxy_model/scale_ablation"
mkdir -p "${ABL}/logs"

WEBHOOK="${DISCORD_WEBHOOK:-https://discord.com/api/webhooks/1500242707522519183/NqqAxuo5n0Ex4X81vMcq7gDOjrP3oAURAVZtgrZT5CramGNfBGPxuEvC2NnxwYBkfs9j}"
_ping() { curl -sS -X POST -H 'Content-Type: application/json' -d "{\"content\": \"$1\"}" "${WEBHOOK}" >/dev/null 2>&1 || true; }
_ping "meanpool-scale 🚀 start · pooled instrument twins for 4 arms + fm-control · job ${SLURM_JOB_ID:-?}"
trap '_ping "meanpool-scale $([ $? -eq 0 ] && echo ✅ || echo ❌) · job ${SLURM_JOB_ID:-?}"' EXIT

source ~/.bashrc
conda activate torchenv
export PYTHONPATH="/home/pablomer/orcd/pool/tess-generative:${PYTHONPATH:-}"

for V in scale-1000 scale-3162 scale-10000 scale-31622 fm-control; do
  H5="${E}/outputs/embeddings/${V}.h5"
  PH5="${E}/outputs/embeddings/${V}-meanpool.h5"
  CSV="${E}/outputs/predictions/predict_${V}-meanpool.csv"

  if [[ ! -f "${H5}" ]]; then
    echo "[skip] ${V}: source H5 missing (${H5}) — its R2 job may still be running"
    continue
  fi
  if [[ ! -f "${PH5}" ]]; then
    echo "[build] ${V}-meanpool"
    python "${E}/build_meanpool_h5.py" --in "${H5}" --out "${PH5}" --tokens 4
  else
    echo "[skip] ${PH5} exists"
  fi
  if [[ ! -f "${CSV}" ]]; then
    echo "[probe] ${V}-meanpool"
    python "${E}/predict_combined.py" --embeddings "${PH5}" --out "${CSV}"
  else
    echo "[skip] ${CSV} exists"
  fi
done

echo "[meanpool-scale] done"
ls -l "${E}"/outputs/predictions/predict_*meanpool*.csv
