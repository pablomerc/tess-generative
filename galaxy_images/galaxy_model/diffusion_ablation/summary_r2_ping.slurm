#!/bin/bash
#SBATCH --job-name=r2-ping
#SBATCH --partition=mit_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=/home/pablomer/orcd/pool/tess-generative/galaxy_images/galaxy_model/diffusion_ablation/logs/%x_%j.out
#SBATCH --error=/home/pablomer/orcd/pool/tess-generative/galaxy_images/galaxy_model/diffusion_ablation/logs/%x_%j.err
#
# FM-vs-DDPM downstream R2 table: results/<stem>.{pdf,tex} + Discord post.
# Submit with --dependency=afterany:<r2 jobs>; checks CSV existence itself.

set -eo pipefail
REPO="/home/pablomer/orcd/pool/tess-generative"
ABL="${REPO}/galaxy_images/galaxy_model/diffusion_ablation"
PRED="${REPO}/galaxy_images/galaxy_model/downstream_evaluation/engaging/outputs/predictions"
FM_GLOB="${FM_GLOB:-predict_fm-control-29k*.csv}"
DDPM_GLOB="${DDPM_GLOB:-predict_diffusion-base-28k*.csv}"
STEM="${STEM:-${ABL}/results/r2_fm_vs_ddpm_28k29k}"
TAG="${TAG:-preview @28k/29k steps}"

WEBHOOK="${DISCORD_WEBHOOK:-https://discord.com/api/webhooks/1500242707522519183/NqqAxuo5n0Ex4X81vMcq7gDOjrP3oAURAVZtgrZT5CramGNfBGPxuEvC2NnxwYBkfs9j}"
_ping() { curl -sS -X POST -H 'Content-Type: application/json' -d "$1" "${WEBHOOK}" >/dev/null 2>&1 || true; }

source ~/.bashrc
conda activate torchenv
cd "${REPO}"
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"

FM_CSV="$(ls -t ${PRED}/${FM_GLOB} 2>/dev/null | head -1 || true)"
DDPM_CSV="$(ls -t ${PRED}/${DDPM_GLOB} 2>/dev/null | head -1 || true)"
if [[ -z "${FM_CSV}" || -z "${DDPM_CSV}" ]]; then
  _ping "{\"content\": \"⚠️ r2-ping: missing predict CSV (fm='${FM_CSV}', ddpm='${DDPM_CSV}') — a downstream job likely failed; check downstream_evaluation/engaging logs.\"}"
  exit 0
fi

PAYLOAD="$(mktemp)"
python -u -m galaxy_images.galaxy_model.diffusion_ablation.make_table_pdf \
  --kind r2 --fm-csv "${FM_CSV}" --ddpm-csv "${DDPM_CSV}" \
  --out-stem "${STEM}" --payload-out "${PAYLOAD}" --tag "${TAG}"
curl -sS -X POST -H 'Content-Type: application/json' -d @"${PAYLOAD}" "${WEBHOOK}" >/dev/null 2>&1 || true
rm -f "${PAYLOAD}"
echo "posted; table at ${STEM}.pdf"
