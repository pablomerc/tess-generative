#!/bin/bash
# One-time wandb setup for the engaging downstream pipeline.
#
# Two modes:
#   1. Online logging (recommended if you want runs in the W&B UI):
#        export WANDB_API_KEY=<your-key>   # grab from https://wandb.ai/authorize
#        bash setup_wandb.sh
#      The script will `wandb login` non-interactively, then flip the slurm
#      scripts from WANDB_MODE=disabled to WANDB_MODE=online.
#
#   2. Disabled (current default — what we used for the smoke):
#        bash setup_wandb.sh --disable
#      Leaves WANDB_MODE=disabled in the slurm scripts (a no-op if that's
#      already the case).

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source ~/.bashrc
conda activate torchenv

MODE="online"
if [[ "${1:-}" == "--disable" ]]; then
    MODE="disabled"
fi

if [[ "${MODE}" == "online" ]]; then
    if [[ -z "${WANDB_API_KEY:-}" ]]; then
        echo "ERROR: WANDB_API_KEY not set."
        echo "Get your key from https://wandb.ai/authorize and run:"
        echo "  export WANDB_API_KEY=<your-key>"
        echo "  bash $0"
        exit 1
    fi
    echo "Logging in to wandb (non-interactive)..."
    wandb login --relogin "${WANDB_API_KEY}"
    echo "wandb status:"
    wandb status || true
fi

echo ""
echo "Setting WANDB_MODE=${MODE} in slurm scripts:"
for f in run_smoke.slurm run_prepare.slurm smoke_test.sh; do
    target="${SCRIPT_DIR}/${f}"
    if grep -q "^export WANDB_MODE=" "${target}" 2>/dev/null; then
        sed -i "s|^export WANDB_MODE=.*|export WANDB_MODE=${MODE}|" "${target}"
        echo "  patched ${f}"
    else
        echo "  (${f} has no WANDB_MODE line — skipping)"
    fi
done

echo ""
echo "Done. Verify:"
grep -nH "^export WANDB_MODE=" "${SCRIPT_DIR}/run_smoke.slurm" "${SCRIPT_DIR}/run_prepare.slurm" "${SCRIPT_DIR}/smoke_test.sh" 2>/dev/null || true
