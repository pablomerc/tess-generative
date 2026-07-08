#!/bin/bash
# Dry-run the discord overlap-check on the smoke-build index.
# Writes the PNG and posts to discord. Override DISCORD_WEBHOOK env var to use
# a different webhook.

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="/home/pablomer/orcd/pool/tess-generative"

source ~/.bashrc
conda activate torchenv
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

DISCORD_WEBHOOK="${DISCORD_WEBHOOK:-https://discord.com/api/webhooks/1500242707522519183/NqqAxuo5n0Ex4X81vMcq7gDOjrP3oAURAVZtgrZT5CramGNfBGPxuEvC2NnxwYBkfs9j}"

python "${SCRIPT_DIR}/check_overlap_visual.py" \
    --index   "${SCRIPT_DIR}/outputs/smoke/index/overlap.parquet" \
    --n       8 \
    --seed    0 \
    --out     "${SCRIPT_DIR}/outputs/smoke/plots/overlap_check.png" \
    --webhook "${DISCORD_WEBHOOK}" \
    --message "smoke overlap check (HSC | Legacy)"
