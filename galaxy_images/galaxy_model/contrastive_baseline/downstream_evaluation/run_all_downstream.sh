#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

"${SCRIPT_DIR}/run_stage1_prepare.sh"
"${SCRIPT_DIR}/run_stage2_predict_plot.sh"

echo "All downstream stages completed."
