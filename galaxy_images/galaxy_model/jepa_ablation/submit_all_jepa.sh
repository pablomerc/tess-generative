#!/bin/bash
# Submit all four JEPA ablation arms (PLAN.md run matrix) to pg_mki_aryeh.
#   bash submit_all_jepa.sh [MAX_STEPS]
set -eo pipefail

JEPA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAX_STEPS="${1:-100000}"
mkdir -p "${JEPA_DIR}/logs"

WEBHOOK="${DISCORD_WEBHOOK:-https://discord.com/api/webhooks/1500242707522519183/NqqAxuo5n0Ex4X81vMcq7gDOjrP3oAURAVZtgrZT5CramGNfBGPxuEvC2NnxwYBkfs9j}"

ARMS=(hsc-resnet hsc-vit cf cross)
SUBMITTED=()
for ARM in "${ARMS[@]}"; do
    JID=$(sbatch --parsable --job-name="jepa-${ARM}" \
                 --export=ALL,ARM=${ARM},MAX_STEPS=${MAX_STEPS} \
                 "${JEPA_DIR}/train_jepa.slurm")
    echo "submitted jepa-${ARM}: job ${JID}"
    SUBMITTED+=("${ARM}=${JID}")
done

curl -sS -X POST -H 'Content-Type: application/json' \
    -d "{\"content\": \"jepa-ablation 📦 submitted 4 training arms (${MAX_STEPS} steps each): ${SUBMITTED[*]}\"}" \
    "${WEBHOOK}" >/dev/null 2>&1 || true
