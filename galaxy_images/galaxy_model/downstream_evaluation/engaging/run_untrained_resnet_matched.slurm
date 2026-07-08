#!/bin/bash
#SBATCH --job-name=engaging-untr-matched
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=2:00:00
#SBATCH --partition=mit_normal_gpu
#SBATCH --mem=32G
#SBATCH --output=/orcd/pool/007/pablomer/tess-generative/galaxy_images/galaxy_model/downstream_evaluation/engaging/logs/%x_%j.out
#SBATCH --error=/orcd/pool/007/pablomer/tess-generative/galaxy_images/galaxy_model/downstream_evaluation/engaging/logs/%x_%j.err
#SBATCH --chdir=/orcd/pool/007/pablomer/tess-generative
#SBATCH --open-mode=append

# Matched-architecture untrained baseline: random-init ResNetEncoder with
# cross_attention_dim=16 (same as base config). Encodes only overlap subset.
# Probes with the standard predict_combined.py loop and pushes a side-by-side
# CSV comparing the seed-0 512-dim baseline to this matched 64-dim baseline.

set -eo pipefail

VARIANT="untrained-resnet-matched"
SCRIPT_DIR="/home/pablomer/orcd/pool/tess-generative/galaxy_images/galaxy_model/downstream_evaluation/engaging"
REPO_ROOT="/home/pablomer/orcd/pool/tess-generative"

DISCORD_WEBHOOK="${DISCORD_WEBHOOK:-https://discord.com/api/webhooks/1500242707522519183/NqqAxuo5n0Ex4X81vMcq7gDOjrP3oAURAVZtgrZT5CramGNfBGPxuEvC2NnxwYBkfs9j}"
_ping() { curl -sS -X POST -H 'Content-Type: application/json' -d "{\"content\": \"$1\"}" "${DISCORD_WEBHOOK}" >/dev/null || true; }
_ping_file() {
    if [[ -f "$2" ]]; then
        curl -sS -F "payload_json={\"content\":\"$1\"}" -F "file=@$2" "${DISCORD_WEBHOOK}" >/dev/null || true
    else
        _ping "$1 (file missing: $2)"
    fi
}
_on_exit() {
    rc=$?
    if [[ $rc -ne 0 ]]; then
        _ping "engaging-${VARIANT} ${SLURM_JOB_ID:-?} ❌ failed (rc=$rc) at $(date '+%F %T')"
    fi
}
trap _on_exit EXIT
_ping "engaging-${VARIANT} ${SLURM_JOB_ID:-?} started at $(date '+%F %T')  (matched (B,4,16) shape, overlap-only)"

source ~/.bashrc
conda activate torchenv
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export TORCH_BLAS_PREFER_HIPBLASLT=0
export OMP_NUM_THREADS=1
export WANDB_MODE=disabled

cd "${REPO_ROOT}"

H5="${SCRIPT_DIR}/outputs/embeddings/${VARIANT}.h5"
CSV="${SCRIPT_DIR}/outputs/predictions/predict_${VARIANT}.csv"
SUMMARY="${SCRIPT_DIR}/outputs/predictions/untrained_resnet_matched_compare.csv"

mkdir -p "${SCRIPT_DIR}/outputs/embeddings" "${SCRIPT_DIR}/outputs/predictions"

if [[ ! -f "${H5}" ]]; then
    python "${SCRIPT_DIR}/prepare_untrained_resnet_matched.py" \
        --index-dir "${SCRIPT_DIR}/outputs/index" \
        --out       "${H5}" \
        --cross-attention-dim 16 \
        --seed-e1   0 \
        --seed-e2   1
else
    echo "[skip] ${H5} already exists"
fi

if [[ ! -f "${CSV}" ]]; then
    python "${SCRIPT_DIR}/predict_combined.py" \
        --embeddings "${H5}" \
        --out        "${CSV}"
else
    echo "[skip] ${CSV} already exists"
fi

# Comparison vs the existing 512-dim untrained baseline.
python - "${CSV}" "${SUMMARY}" <<'PY'
import sys, pandas as pd
new_csv, out_csv = sys.argv[1], sys.argv[2]
old = pd.read_csv("/orcd/pool/007/pablomer/tess-generative/galaxy_images/galaxy_model/downstream_evaluation/engaging/outputs/predictions/predict_untrained-resnet.csv")
new = pd.read_csv(new_csv)
key = ["task_family", "target", "latent_variant", "score_metric"]
m = old[key + ["score"]].merge(new[key + ["score"]], on=key, suffixes=("_512dim", "_matched"))
m["delta"] = m["score_matched"] - m["score_512dim"]
m = m.sort_values(["task_family", "score_512dim"], ascending=[True, False])
m.to_csv(out_csv, index=False)
print(f"\nWrote {out_csv}  ({len(m)} rows)")
print(m.to_string(index=False, float_format=lambda x: f"{x: .3f}"))
PY

_ping_file "engaging-${VARIANT} ${SLURM_JOB_ID:-?} ✅ done — 512-dim untrained vs matched-(B,4,16) untrained" "${SUMMARY}"

echo "Done at $(date)"
