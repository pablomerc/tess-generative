#!/usr/bin/env bash
# Driver: submits one sbatch per variant in variants.json (prepare -> probe -> plot -> discord).
# Idempotent — skips variants whose checkpoint isn't on disk yet, and the per-variant
# slurm skips embeddings/predictions already cached.
#
# Usage (from any directory):
#   bash /orcd/pool/007/pablomer/tess-generative/galaxy_images/galaxy_model/downstream_evaluation/engaging/run_per_variant.sh
#
# Optional env:
#   ONLY=base-15NB        # restrict to a single variant (handy for smoke tests)
#   DRY_RUN=1             # print the sbatch commands instead of submitting

set -eo pipefail

SCRIPT_DIR="/home/pablomer/orcd/pool/tess-generative/galaxy_images/galaxy_model/downstream_evaluation/engaging"
VARIANTS_JSON="${SCRIPT_DIR}/variants.json"
UNTR_CSV="${SCRIPT_DIR}/outputs/predictions/predict_untrained-resnet.csv"

# 1. Untrained-ResNet baseline (one-off; needed by every plot).
DEPENDENCY=""
if [[ ! -f "${UNTR_CSV}" ]]; then
    if [[ "${DRY_RUN:-0}" == "1" ]]; then
        echo "[dry] sbatch ${SCRIPT_DIR}/run_untrained_resnet.slurm"
        UNT_JOBID="DRY"
    else
        UNT_JOBID="$(sbatch --parsable "${SCRIPT_DIR}/run_untrained_resnet.slurm")"
        echo "Submitted untrained-resnet baseline as job ${UNT_JOBID}"
    fi
    DEPENDENCY="--dependency=afterok:${UNT_JOBID}"
else
    echo "[skip] untrained-resnet CSV already present at ${UNTR_CSV}"
fi

# 2. Iterate variants. Resolve checkpoint via colon-separated globs; first hit wins.
python3 - "${VARIANTS_JSON}" "${ONLY:-}" <<'PY' | while IFS=$'\t' read -r NAME CKPT MODULE MODEL_CLASS; do
import glob, json, os, sys

variants_json, only = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else "")
data = json.load(open(variants_json))
for v in data["variants"]:
    if only and v["name"] != only:
        continue
    found = ""
    for pat in v["ckpt_glob"].split(":"):
        hits = sorted(glob.glob(pat))
        if hits:
            found = hits[0]
            break
    if not found:
        print(f"# missing ckpt for {v['name']} (tried {v['ckpt_glob']})", file=sys.stderr)
        continue
    print(f"{v['name']}\t{found}\t{v['module']}\t{v['model_class']}")
PY
    if [[ -z "${NAME}" ]]; then continue; fi
    EXPORTS="ALL,VARIANT=${NAME},CKPT=${CKPT},MODULE=${MODULE},MODEL_CLASS=${MODEL_CLASS}"
    if [[ "${DRY_RUN:-0}" == "1" ]]; then
        echo "[dry] sbatch ${DEPENDENCY} --export=${EXPORTS} --job-name=engaging-${NAME} ${SCRIPT_DIR}/run_variant.slurm"
    else
        JOBID="$(sbatch --parsable ${DEPENDENCY} --export="${EXPORTS}" --job-name="engaging-${NAME}" "${SCRIPT_DIR}/run_variant.slurm")"
        echo "Submitted ${NAME} as job ${JOBID}  (ckpt=${CKPT})"
    fi
done

echo "All submissions complete."
