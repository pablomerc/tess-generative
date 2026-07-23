#!/bin/bash
#
# Submit the full 2x2 contrastive experiment matrix on Engaging:
#
#   neighbors ∈ {spatial (random_neighbors=0), random (random_neighbors=1)}
#   pooling   ∈ {avg, conv1x1}
#
# Each training job self-chains to MAX_STEPS across the 6h wall limit and, on
# completion, launches its own downstream eval (which posts an "Ours vs
# Contrastive" R² plot to Discord). The eval that finishes last also renders the
# combined 4-variant comparison.
#
# Usage:
#   bash submit_all_engaging.sh              # 100k steps (default)
#   MAX_STEPS=300000 bash submit_all_engaging.sh
#   DRY_RUN=1 bash submit_all_engaging.sh    # print sbatch commands only

set -eo pipefail

CB_DIR="/home/pablomer/orcd/pool/tess-generative/galaxy_images/galaxy_model/contrastive_baseline"
MAX_STEPS="${MAX_STEPS:-100000}"
WEBHOOK="https://discord.com/api/webhooks/1500242707522519183/NqqAxuo5n0Ex4X81vMcq7gDOjrP3oAURAVZtgrZT5CramGNfBGPxuEvC2NnxwYBkfs9j"

mkdir -p "${CB_DIR}/logs"

# tag                 encoder_pool   random_neighbors
MATRIX=(
    "spatial-avg      avg            0"
    "spatial-conv1x1  conv1x1        0"
    "random-avg       avg            1"
    "random-conv1x1   conv1x1        1"
)

submitted=()
for row in "${MATRIX[@]}"; do
    read -r TAG POOL RND <<< "${row}"
    CMD=(sbatch --job-name="contr-train-${TAG}"
         --export=ALL,RUN_TAG=${TAG},ENCODER_POOL=${POOL},RANDOM_NEIGHBORS=${RND},MAX_STEPS=${MAX_STEPS}
         "${CB_DIR}/train_contrastive_engaging.slurm")
    echo "+ ${CMD[*]}"
    if [[ "${DRY_RUN:-0}" == "1" ]]; then
        continue
    fi
    out=$("${CMD[@]}")
    echo "  ${out}"
    submitted+=("${TAG}=${out##* }")
done

if [[ "${DRY_RUN:-0}" != "1" ]]; then
    msg="🚀 Submitted 4 contrastive runs on Engaging (${MAX_STEPS} steps each): ${submitted[*]}. Each will auto-eval + post an Ours-vs-Contrastive R² plot when done."
    curl -sS -X POST -H 'Content-Type: application/json' -d "{\"content\": \"${msg}\"}" "${WEBHOOK}" >/dev/null 2>&1 || true
    echo "${msg}"
fi
