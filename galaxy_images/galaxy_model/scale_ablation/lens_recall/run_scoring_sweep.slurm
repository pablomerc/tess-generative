#!/bin/bash
#SBATCH --job-name=lens-decks
#SBATCH --partition=pg_mki_aryeh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=/home/pablomer/orcd/pool/tess-generative/galaxy_images/galaxy_model/scale_ablation/lens_recall/outputs/%x_%j.out
#SBATCH --error=/home/pablomer/orcd/pool/tess-generative/galaxy_images/galaxy_model/scale_ablation/lens_recall/outputs/%x_%j.err
#
# Metric 1 (lens DISCOVERY, human-scored): encode 10 checkpoints over the shared gallery,
# then build blinded scoring decks for two queries and post them to Discord.
#
# Runs on pg_mki_aryeh deliberately: mit_normal_gpu nodes read Ceph so slowly that a 2.2 GB
# checkpoint load stalled for 22 min there (job 18933414), while node4900 is fine.

set -eo pipefail
REPO="/home/pablomer/orcd/pool/tess-generative"
LR="${REPO}/galaxy_images/galaxy_model/scale_ablation/lens_recall"
SA="/orcd/pool/007/pablomer/checkpoints_new/scale_ablation"
SNAP="/orcd/data/mki_aryeh/001/pablomer/scale_ablation/snapshots"
DIFF="/orcd/pool/007/pablomer/checkpoints_new/diffusion_ablation/frozen"
mkdir -p "${LR}/outputs" "${LR}/latents" "${LR}/decks"

WEBHOOK="${DISCORD_WEBHOOK:-https://discord.com/api/webhooks/1500242707522519183/NqqAxuo5n0Ex4X81vMcq7gDOjrP3oAURAVZtgrZT5CramGNfBGPxuEvC2NnxwYBkfs9j}"
_ping() { curl -sS -X POST -H 'Content-Type: application/json' -d "{\"content\": \"$1\"}" "${WEBHOOK}" >/dev/null 2>&1 || true; }
_pingf() { curl -sS -F "payload_json={\"content\": \"$1\"}" -F "file=@$2" "${WEBHOOK}" >/dev/null 2>&1 || true; }
_ping "lens-decks 🔭 start · 10 checkpoints + 2 instrument controls · job ${SLURM_JOB_ID:-?}"
trap '_ping "lens-decks $([ $? -eq 0 ] && echo ✅ || echo ❌) · job ${SLURM_JOB_ID:-?}"' EXIT

source ~/.bashrc
conda activate torchenv
cd "${LR}"
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

BM1000=$(ls ${SNAP}/scale_1000/bestmse_*.ckpt   | head -1)
BM3162=$(ls ${SNAP}/scale_3162/bestmse_*.ckpt   | head -1)
BM10K=$(ls  ${SNAP}/scale_10000/bestmse_*.ckpt  | head -1)
BM31K=$(ls  ${SNAP}/scale_31622/bestmse_*.ckpt  | head -1)

srun --unbuffered python -u encode_many.py --out-dir "${LR}/latents" --source ram48 \
  --checkpoint "scale-1000-75k:${SA}/frozen/scale_1000/scale-1000-75k.ckpt" \
  --checkpoint "scale-3162-75k:${SA}/frozen/scale_3162/scale-3162-75k.ckpt" \
  --checkpoint "scale-10000-75k:${SA}/frozen/scale_10000/scale-10000-75k.ckpt" \
  --checkpoint "scale-31622-75k:${SA}/frozen/scale_31622/scale-31622-75k.ckpt" \
  --checkpoint "scale-1000-bestmse:${BM1000}" \
  --checkpoint "scale-3162-bestmse:${BM3162}" \
  --checkpoint "scale-10000-bestmse:${BM10K}" \
  --checkpoint "scale-31622-bestmse:${BM31K}" \
  --checkpoint "base-87k:/orcd/pool/007/pablomer/checkpoints_new/base/snapshot.ckpt" \
  --checkpoint "fm-control-75k:${DIFF}/fm-control-75k-fixed.ckpt"

for Q in lens48 lens33; do
  srun --unbuffered python -u make_scoring_deck.py \
    --latents-dir "${LR}/latents" --out-dir "${LR}/decks" --query "${Q}" --k 8
  N=$(python -c "import json;print(len(json.load(open('${LR}/decks/deck_${Q}_KEY.json'))['rows']))")
  _ping "**Lens-discovery scoring — query \`${Q}\`** · ${N} rows, shuffled & anonymised (2 are instrument-space controls; you are not told which). For each row reply with the ranks that look like plausible strong lenses, e.g. \`A: 1,3,4\` / \`B: none\`. Rank 1 = closest neighbour; the query's own cross-survey counterpart is excluded."
  for P in "${LR}/decks/deck_${Q}"_page*.png; do
    _pingf "\`${Q}\` — $(basename "${P}")" "${P}"
  done
done

_ping "lens-decks 📋 decks posted. Keys held at \`decks/deck_*_KEY.json\` (not posted) — I decode after you score."
