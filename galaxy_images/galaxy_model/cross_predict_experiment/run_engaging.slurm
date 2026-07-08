#!/bin/bash
#SBATCH --job-name=engaging-xpred
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --partition=mit_normal_gpu
#SBATCH --mem=32G
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --open-mode=append

# Cross-prediction baseline: train ResNet18 from scratch on one survey's image
# to predict the OTHER survey's instrument labels (overlap subset = MMU rows).
# Submit twice in parallel, once per direction:
#
#   sbatch --export=ALL,XPRED_DIRECTION=hsc_to_legacy run_engaging.slurm
#   sbatch --export=ALL,XPRED_DIRECTION=legacy_to_hsc run_engaging.slurm
#
# Writes outputs/r2_results_<direction>.csv. The accompanying comparison plot
# is skipped on engaging (BASELINE_CSV from the MIT pipeline isn't here).

set -eo pipefail

DIRECTION="${XPRED_DIRECTION:-hsc_to_legacy}"
EPOCHS="${XPRED_EPOCHS:-50}"
BATCH_SIZE="${XPRED_BATCH_SIZE:-64}"
MAX_SAMPLES="${XPRED_MAX_SAMPLES:-}"   # empty = use full dataset

SCRIPT_DIR="/home/pablomer/orcd/pool/tess-generative/galaxy_images/galaxy_model/cross_predict_experiment"
REPO_ROOT="/home/pablomer/orcd/pool/tess-generative"

DISCORD_WEBHOOK="${DISCORD_WEBHOOK:-https://discord.com/api/webhooks/1500242707522519183/NqqAxuo5n0Ex4X81vMcq7gDOjrP3oAURAVZtgrZT5CramGNfBGPxuEvC2NnxwYBkfs9j}"
_ping() { curl -sS -X POST -H 'Content-Type: application/json' -d "{\"content\": $(python -c 'import json,sys; print(json.dumps(sys.argv[1]))' "$1")}" "${DISCORD_WEBHOOK}" >/dev/null || true; }
_ping_file() {
    if [[ -f "$2" ]]; then
        curl -sS -F "payload_json={\"content\":$(python -c 'import json,sys; print(json.dumps(sys.argv[1]))' "$1")}" -F "file=@$2" "${DISCORD_WEBHOOK}" >/dev/null || true
    else
        _ping "$1 (file missing: $2)"
    fi
}
_on_exit() {
    rc=$?
    if [[ $rc -ne 0 ]]; then
        _ping "engaging-xpred ${DIRECTION} ${SLURM_JOB_ID:-?} ❌ failed (rc=$rc) at $(date '+%F %T')"
    fi
}
trap _on_exit EXIT

source ~/.bashrc
conda activate torchenv
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export TORCH_BLAS_PREFER_HIPBLASLT=0
export OMP_NUM_THREADS=4

cd "${REPO_ROOT}"

OUT_DIR="${SCRIPT_DIR}"
RESULT_CSV="${OUT_DIR}/r2_results_${DIRECTION}.csv"

MAX_SAMPLES_ARG=""
if [[ -n "${MAX_SAMPLES}" ]]; then
    MAX_SAMPLES_ARG="--max-samples ${MAX_SAMPLES}"
fi

_ping "engaging-xpred ${DIRECTION} ${SLURM_JOB_ID:-?} 🚀 started — epochs=${EPOCHS}, batch=${BATCH_SIZE}${MAX_SAMPLES:+, max_samples=${MAX_SAMPLES}}, hdf5=/orcd/pool/007/pablomer/neighbours_v2.h5 (in-script Discord pings every 5 epochs)"

python -u "${SCRIPT_DIR}/cross_predict.py" \
    --direction   "${DIRECTION}" \
    --epochs      "${EPOCHS}" \
    --batch-size  "${BATCH_SIZE}" \
    --output-dir  "${OUT_DIR}" \
    --no-wandb \
    ${MAX_SAMPLES_ARG}

echo ""
echo "=== R² CSV ==="
cat "${RESULT_CSV}"

SUMMARY_TXT="$(python - <<PY
import pandas as pd
df = pd.read_csv("${RESULT_CSV}")
lines = ["cross-predict ${DIRECTION} R² (best validation epoch):"]
for _, row in df.iterrows():
    lines.append("  {:40s}  R² = {:+.4f}".format(row["target"], row["r2_cross_predict"]))
lines.append("  {:40s}  R² = {:+.4f}".format("AVERAGE", df["r2_cross_predict"].mean()))
print("\n".join(lines))
PY
)"

echo ""
echo "=== Summary ==="
echo "${SUMMARY_TXT}"

_ping_file "engaging-xpred ${DIRECTION} ${SLURM_JOB_ID:-?} ✅ done at $(date '+%F %T')\n\`\`\`\n${SUMMARY_TXT}\n\`\`\`" "${RESULT_CSV}"

echo "Done at $(date)"
