#!/bin/bash
#SBATCH --job-name=engaging-instr2phys-v2
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:30:00
#SBATCH --partition=mit_normal_gpu
#SBATCH --mem=16G
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --open-mode=append

# v2 metadata-only baseline: same probe as run_instrument_to_physics.slurm but
# with the audit-driven preprocessing (--preprocess v2):
#   - drop HSC variance cols + duplicate legacy_psf_fwhm + duplicate legacy_ra/dec
#   - log10 PSFDEPTH/GALDEPTH, log1p NOBS, -2.5*log10 MW_TRANSMISSION, z-score all
#   - log-space training for provabgs_avg_sfr and provabgs_z_mw (R² reported on
#     original linear scale)
# Writes outputs/predictions/predict_instrument_baseline_v2.csv (does NOT
# overwrite the v1 CSV).

set -eo pipefail

SCRIPT_DIR="/home/pablomer/orcd/pool/tess-generative/galaxy_images/galaxy_model/downstream_evaluation/engaging"
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
        _ping "engaging-instr2phys-v2 ${SLURM_JOB_ID:-?} ❌ failed (rc=$rc) at $(date '+%F %T')"
    fi
}
trap _on_exit EXIT

source ~/.bashrc
conda activate torchenv
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export TORCH_BLAS_PREFER_HIPBLASLT=0
export OMP_NUM_THREADS=1

cd "${REPO_ROOT}"

mkdir -p "${SCRIPT_DIR}/outputs/predictions"

LABELS_SRC="${SCRIPT_DIR}/outputs/embeddings/base.h5"
OUT_CSV="${SCRIPT_DIR}/outputs/predictions/predict_instrument_baseline_v2.csv"
SEEDS="0,1,2,3,4"
MAX_EPOCHS=200

_ping "engaging-instr2phys-v2 ${SLURM_JOB_ID:-?} 🚀 started — seeds=${SEEDS}, max_epochs=${MAX_EPOCHS}, --preprocess v2 (audit-driven log/std)"

python "${SCRIPT_DIR}/predict_instrument_to_physics.py" \
    --labels-source "${LABELS_SRC}" \
    --out           "${OUT_CSV}" \
    --seeds         "${SEEDS}" \
    --max-epochs    "${MAX_EPOCHS}" \
    --preprocess    v2 \
    --device        cuda

echo "=== Raw CSV ==="
cat "${OUT_CSV}"

SUMMARY_TXT="$(python - <<PY
import pandas as pd
df_v2 = pd.read_csv("${OUT_CSV}")
order = ["desi_z","provabgs_logmstar","provabgs_tage_mw","provabgs_avg_sfr","provabgs_z_mw","ssfr"]
agg2 = df_v2.groupby("target")["score"].agg(["mean","std","count"]).reindex(order).dropna(how="all")

# v1 reference (one row per target, no seed col).
v1_path = "${SCRIPT_DIR}/outputs/predictions/predict_instrument_baseline.csv"
import os
if os.path.exists(v1_path):
    df_v1 = pd.read_csv(v1_path)
    v1_by_target = df_v1.set_index("target")["score"]
else:
    v1_by_target = pd.Series(dtype=float)

lines = ["instrument → physics R² (v2, mean ± std across {} seeds, n_test={}):".format(
    int(agg2["count"].iloc[0]), int(df_v2["n_test"].iloc[0]))]
lines.append("  {:24s}  {:>12s}    {:>12s}".format("target", "v1", "v2"))
for tgt, row in agg2.iterrows():
    v1 = v1_by_target.get(tgt, float("nan"))
    v1s = "  {:+.4f}".format(v1) if v1 == v1 else "       n/a "
    v2s = "{:+.4f} ± {:.4f}".format(row["mean"], row["std"])
    lines.append("  {:24s}  {:>12s}    {}".format(tgt, v1s, v2s))
print("\n".join(lines))
PY
)"

echo ""
echo "=== Summary ==="
echo "${SUMMARY_TXT}"

_ping_file "engaging-instr2phys-v2 ${SLURM_JOB_ID:-?} ✅ done at $(date '+%F %T')\n\`\`\`\n${SUMMARY_TXT}\n\`\`\`" "${OUT_CSV}"

echo "Done at $(date)"
