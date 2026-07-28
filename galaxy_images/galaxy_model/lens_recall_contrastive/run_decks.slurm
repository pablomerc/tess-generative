#!/bin/bash
#SBATCH --job-name=clens-deck
#SBATCH --partition=mit_normal_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/home/pablomer/orcd/pool/tess-generative/galaxy_images/galaxy_model/lens_recall_contrastive/outputs/%x_%j.out
#SBATCH --error=/home/pablomer/orcd/pool/tess-generative/galaxy_images/galaxy_model/lens_recall_contrastive/outputs/%x_%j.err
#
# Blinded 4-row Fig-5 deck: contrastive conv1x1 vs its declared flow-matching counterparts.
#
# The 2x2 being probed (rows shuffled before posting, so this ordering is not the deck ordering):
#
#                     |  saw the lenses (unlabeled)      |  never saw them (zero-shot)
#   ------------------+----------------------------------+---------------------------------
#   generative (FM)   |  base-87k  (the paper model)     |  base-5NB-DWNVAL
#   contrastive 1x1   |  contrastive-spatial-conv1x1     |  contrastive-spatial-conv1x1-DWNVAL
#
# 73 of the 74 catalog lenses -- INCLUDING this query -- are in the DWNVAL holdout, so the two
# right-hand arms never saw any of them. The pairings are the ones declared in
# contrastive_ablation/evaluation/evaluation_config.json (primary_pair / secondary_pair).
#
# Partition choice: the scale sweep used pg_mki_aryeh because mit_normal_gpu read POOL Ceph so
# slowly that a 2.2 GB checkpoint load stalled 22 min there (job 18933414). Two things changed:
#   1. pg_mki_aryeh/node4900 is fully allocated by our own training runs (cf-fm-long, scale-100)
#      and we are not going to evict them three days before the rebuttal deadline;
#   2. the 17 GB RAM48 store now also lives on mki_aryeh group storage, which is a different
#      filesystem (hstor003-n1:/group) and is 1% used, so the big read no longer touches Ceph.
# What remains on Ceph is ~2.7 GB of checkpoints and a few MB of neighbours_v2 metadata/stamps.
# This job is I/O-bound, not compute-bound: the actual GPU work is ~20 s/arm (ResNet18 on 48x48),
# so an L40S is ample and leaves the H200s for training. 4 h wall is slack for slow reads; the
# encode step is idempotent (it skips any arm whose latents already exist), so a timeout resumes.

set -eo pipefail
REPO="/home/pablomer/orcd/pool/tess-generative"
GM="${REPO}/galaxy_images/galaxy_model"
CLR="${GM}/lens_recall_contrastive"
SLR="${GM}/scale_ablation/lens_recall"
CKPT="/orcd/pool/007/pablomer/checkpoints_new"
INC="/home/pablomer/orcd/pool/contrastive_ablation_checkpoints"
mkdir -p "${CLR}/outputs" "${CLR}/decks"

WEBHOOK="${DISCORD_WEBHOOK:-https://discord.com/api/webhooks/1500242707522519183/NqqAxuo5n0Ex4X81vMcq7gDOjrP3oAURAVZtgrZT5CramGNfBGPxuEvC2NnxwYBkfs9j}"
_ping()  { curl -sS -X POST -H 'Content-Type: application/json' -d "{\"content\": \"$1\"}" "${WEBHOOK}" >/dev/null 2>&1 || true; }
_pingf() { curl -sS -F "payload_json={\"content\": \"$1\"}" -F "file=@$2" "${WEBHOOK}" >/dev/null 2>&1 || true; }
_ping "clens-deck 🔭 start · contrastive conv1x1 vs FM, 4 blinded rows on lens48 · job ${SLURM_JOB_ID:-?}"
trap '_ping "clens-deck $([ $? -eq 0 ] && echo ✅ || echo ❌) · job ${SLURM_JOB_ID:-?}"' EXIT

source ~/.bashrc
conda activate torchenv
cd "${CLR}"
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MPLCONFIGDIR=/tmp/matplotlib

# Prefer the group-storage copy of the 17 GB store (fast, uncontended); fall back to POOL if the
# mount is absent on whichever node we land on. Both copies are byte-size identical.
RAM48_GROUP="/orcd/data/mki_aryeh/001/pablomer/efficient_neighs_48"
RAM48_POOL="/orcd/pool/007/pablomer/efficient_neighs_48"
if [ -r "${RAM48_GROUP}/meta.json" ]; then RAM48="${RAM48_GROUP}"; else RAM48="${RAM48_POOL}"; fi
echo "[run] node=$(hostname) ram48=${RAM48}"

# --- encode the three arms not already on disk ------------------------------------------
# base-87k is reused from the scale sweep's cache: it is the SAME latents file behind the row
# already scored in deck_lens48, so its number stays consistent across the two sessions.
# Checkpoint choices are the canonical ones recorded in the downstream embeddings' attrs.
srun --unbuffered python -u encode_arms.py --out-dir "${CLR}/latents" --source ram48 \
  --ram48-dir "${RAM48}" \
  --arm "contrastive-spatial-conv1x1:contrastive_baseline/dual_encoder_contrastive.py:DualEncoderContrastiveModule:${CKPT}/contrastive-spatial-conv1x1/best-epoch=53-step=83000.ckpt" \
  --arm "contrastive-spatial-conv1x1-DWNVAL:contrastive_ablation/dual_encoder_contrastive_variants.py:ContrastiveNegativeAblationModule:${INC}/contrastive-spatial-conv1x1-DWNVAL/best-epoch=48-step=70000.ckpt" \
  --arm "base-5NB-DWNVAL:double_train_fm_neighbors.py:ConditionalFlowMatchingModule:${CKPT}/base-5NB-DWNVAL/best-epoch=197-step=75000.ckpt"

# --- build the blinded deck -------------------------------------------------------------
srun --unbuffered python -u make_deck.py \
  --latents-dir "${CLR}/latents" --latents-dir "${SLR}/latents" \
  --out-dir "${CLR}/decks" --query lens48 --k 8 --rows-per-page 4 \
  --out-prefix deck_lens48_contrastive \
  --note "4-row contrastive-vs-FM 2x2. base-87k physics was previously shown as row D of scale_ablation/lens_recall/decks/deck_lens48 (scored 7/8), so its score here is NOT independent of that session." \
  --arm "contrastive-spatial-conv1x1:physics" \
  --arm "contrastive-spatial-conv1x1-DWNVAL:physics" \
  --arm "base-5NB-DWNVAL:physics" \
  --arm "base-87k:physics"

N=$(python -c "import json;print(len(json.load(open('${CLR}/decks/deck_lens48_contrastive_KEY.json'))['rows']))")
_ping "**Blinded lens retrieval — query \`lens48\`** · ${N} rows, shuffled & anonymised. All rows are the **physics** latent; which model produced each row is withheld until you have scored. For each row reply with the ranks that look like plausible strong lenses, e.g. \`A: 1,3,4\` / \`B: none\`. Rank 1 = closest neighbour; the query's own cross-survey counterpart is excluded."
for P in "${CLR}/decks/deck_lens48_contrastive"*.png; do
  _pingf "\`lens48\` — $(basename "${P}")" "${P}"
done
_ping "clens-deck 📋 posted. Key held at \`decks/deck_lens48_contrastive_KEY.json\` (not posted) — reply with your scores and I decode."
