#!/bin/bash
#SBATCH --job-name=jepa-replot160
#SBATCH --partition=pg_mki_aryeh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/orcd/pool/007/pablomer/tess-generative/galaxy_images/galaxy_model/jepa_ablation/logs/%x_%j.out
#SBATCH --error=/orcd/pool/007/pablomer/tess-generative/galaxy_images/galaxy_model/jepa_ablation/logs/%x_%j.err
#SBATCH --open-mode=append
#
# Re-render the top-anomaly grids at FULL 160x160 resolution (plot-only, CPU) for
# every tag whose scores H5 already exists. plot_anomalies.py --stamp-store full160
# reads the original 160x160 store with 16-thread prefetch; output goes to
# figures_<tag>_160 so the 48x48 grids are kept. Posts each joint panel to Discord.
#
#   sbatch replot160.slurm                       # all available tags
#   TAGS="jepa-cf ours-fmbase" sbatch replot160.slurm

set -eo pipefail
REPO=/orcd/pool/007/pablomer/tess-generative
SRC="${REPO}/galaxy_images/galaxy_model/contrastive_anomalies"
OUT=/orcd/pool/007/pablomer/outputs/contrastive_anomalies

WEBHOOK="${DISCORD_WEBHOOK:-https://discord.com/api/webhooks/1500242707522519183/NqqAxuo5n0Ex4X81vMcq7gDOjrP3oAURAVZtgrZT5CramGNfBGPxuEvC2NnxwYBkfs9j}"
_ping() { curl -sS -X POST -H 'Content-Type: application/json' -d "{\"content\": \"$1\"}" "${WEBHOOK}" >/dev/null 2>&1 || true; }
_post_file() { # $1 message, $2 file
  curl -sS -F "payload_json={\"content\": \"$1\"}" -F "files[0]=@$2" "${WEBHOOK}" >/dev/null 2>&1 || true;
}
_on_exit() { rc=$?; if [[ $rc -ne 0 ]]; then _ping "replot160 ❌ failed (rc=$rc) job ${SLURM_JOB_ID:-?}"; fi; exit $rc; }
trap _on_exit EXIT

source ~/.bashrc
conda activate torchenv
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1
cd "${SRC}"

# Default: every tag with a scores file — EXCEPT the 474k-corpus reruns, whose H5
# schema differs (no row_idx) and which index a different store anyway.
if [[ -z "${TAGS:-}" ]]; then
    TAGS=$(ls "${OUT}"/scores_*.h5 2>/dev/null | sed 's|.*/scores_||; s|\.h5$||' | grep -v "474k" | tr '\n' ' ')
fi
echo "[replot160] tags: ${TAGS}"
_ping "replot160 🚀 re-rendering top-anomaly grids at 160x160 for: ${TAGS} (job ${SLURM_JOB_ID:-?})"

for TAG in ${TAGS}; do
    SCORES="${OUT}/scores_${TAG}.h5"
    [[ -f "${SCORES}" ]] || { echo "[skip] ${TAG}: no scores"; continue; }
    FIGDIR="${OUT}/figures_${TAG}_160"
    echo "=== ${TAG} -> ${FIGDIR} ==="
    python plot_anomalies.py \
        --scores "${SCORES}" \
        --stamp-store full160 \
        --out-dir "${FIGDIR}" \
        --skip-individual
    # Post the joint physics-vs-instrument panel (fall back to any top grid).
    PANEL=$(ls -t "${FIGDIR}"/joint_*.png 2>/dev/null | head -1 || true)
    [[ -z "${PANEL}" ]] && PANEL=$(ls -t "${FIGDIR}"/*.png 2>/dev/null | head -1 || true)
    if [[ -n "${PANEL}" ]]; then
        _post_file "🔭 ${TAG} — top anomalies at 160x160" "${PANEL}"
    fi
done

_ping "replot160 ✅ done — figures in ${OUT}/figures_<tag>_160/"
echo "Done at $(date)"
