#!/bin/bash
#SBATCH --job-name=adv-eval
#SBATCH --partition=mit_normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/orcd/pool/007/pablomer/tess-generative/galaxy_images/galaxy_model/contrastive_ablation/logs/%x-%j.out
#SBATCH --error=/orcd/pool/007/pablomer/tess-generative/galaxy_images/galaxy_model/contrastive_ablation/logs/%x-%j.err
#
# Stratified evaluation of delta_r2 OOF predictions on the FROZEN quality manifest.
# CPU-only: object-paired bootstrap over strata x axes; too slow for a login node.
#
# Submit:
#   sbatch run_adversarial_eval.slurm
# Override the arm pair with OURS= / CONTRASTIVE= (model dir names under delta_r2/outputs).

set -euo pipefail

HERE="${HERE:-/orcd/pool/007/pablomer/tess-generative/galaxy_images/galaxy_model/contrastive_ablation/evaluation/adversarial_testset}"
D="${D:-${HERE}/../delta_r2/outputs}"
PYTHON="${PYTHON:-/orcd/home/002/pablomer/miniforge3/envs/torchenv/bin/python}"

OURS="${OURS:-base-5NB-DWNVAL}"
CONTRASTIVE="${CONTRASTIVE:-contrastive-spatial-conv1x1-DWNVAL}"
OUT_DIR="${OUT_DIR:-${HERE}/outputs/adversarial_eval}"
REPS="${REPS:-2000}"
SEED="${SEED:-0}"

WEBHOOK="${DISCORD_WEBHOOK:-}"
_ping() { [[ -z "${WEBHOOK}" ]] && return 0; curl -s -X POST -H 'Content-Type: application/json' \
    -d "{\"content\": $(printf '%s' "$1" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))')}" \
    "${WEBHOOK}" > /dev/null || true; }

trap '_ping "❌ adv-eval ${SLURM_JOB_ID:-} failed (see logs)"' ERR

cd "${HERE}"
_ping "▶️ adv-eval ${SLURM_JOB_ID:-} started — ${OURS} vs ${CONTRASTIVE}, ${REPS} bootstrap reps."

"${PYTHON}" evaluate_adversarial.py \
  --manifest outputs/adversarial_manifest.csv \
  --pred "ours:hsc=${D}/${OURS}/oof_predictions.csv" \
  --pred "ours:legacy=${D}/${OURS}/oof_predictions.csv" \
  --pred "ours:combined=${D}/${OURS}/oof_predictions.csv" \
  --pred "contrastive:hsc=${D}/${CONTRASTIVE}/oof_predictions.csv" \
  --pred "contrastive:legacy=${D}/${CONTRASTIVE}/oof_predictions.csv" \
  --pred "contrastive:combined=${D}/${CONTRASTIVE}/oof_predictions.csv" \
  --out-dir "${OUT_DIR}" \
  --bootstrap-replicates "${REPS}" \
  --seed "${SEED}"

_ping "✅ adv-eval ${SLURM_JOB_ID:-} done — ${OUT_DIR}"
echo "Done at $(date)"
