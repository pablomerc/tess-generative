#!/bin/bash
#SBATCH --job-name=neighbors_160
#SBATCH --partition=mi3001x
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=%x_%j.log

set -e

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GALAXY_MODEL_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

LATENTS="${SCRIPT_DIR}/../neighbor_latents_2026-04-05.h5"
NEIGHBORS_H5="/work1/jeroenaudenaert/pablomer/data/neighbours_v2.h5"
BATCH_START=0
BATCH_END=7
DISCORD_WEBHOOK="https://discord.com/api/webhooks/1488692651334177071/8b8KvACfQIVYCNY3ovee04BixCEWiqbqp1iQk4z9sXHlgR29kMkGIjl1pahV5uEPSbxe"

CSV_DIR="${SCRIPT_DIR}/../query_results"
OUT_DIR="${SCRIPT_DIR}/query_results"

# Limit OpenBLAS threads to avoid segfault during kNN search
export OPENBLAS_NUM_THREADS=4
export OMP_NUM_THREADS=4

cd "${GALAXY_MODEL_DIR}"

echo "================================================================"
echo "Step 1: kNN search (indices ${BATCH_START}–${BATCH_END})"
echo "================================================================"
python neighbor_search/search_neighbors.py \
    --latents "${LATENTS}" \
    --neighbors-h5 "${NEIGHBORS_H5}" \
    --batch-start "${BATCH_START}" \
    --batch-end "${BATCH_END}" \
    --out "${CSV_DIR}/"

echo "================================================================"
echo "Step 2: Plot 160x160 neighbor grids"
echo "================================================================"
for i in $(seq "${BATCH_START}" "${BATCH_END}"); do
    python neighbor_search/neighbors_original_img/neighbors_plot_160.py \
        --query-idx "${i}" \
        --neighbors-h5 "${NEIGHBORS_H5}"
    echo "Plotted query_${i}"
done

echo "================================================================"
echo "Step 3: Send plots to Discord"
echo "================================================================"
for i in $(seq "${BATCH_START}" "${BATCH_END}"); do
    f="${OUT_DIR}/query_${i}_160.png"
    echo "Sending ${f} ..."
    curl -s \
        -F "file=@${f}" \
        -F "payload_json={\"content\":\"query_${i} — 160x160 neighbor plot\"}" \
        "${DISCORD_WEBHOOK}" \
        | python3 -c "import sys,json; r=json.load(sys.stdin); print('ok' if 'id' in r else r)"
done

echo "================================================================"
echo "Done. ${BATCH_END} - ${BATCH_START} + 1 plots sent to Discord."
echo "================================================================"
