#!/bin/bash
#SBATCH -J lens-neighbor-search-snapshot
#SBATCH -o /work1/jeroenaudenaert/pablomer/logs/job.%j.out
#SBATCH -e /work1/jeroenaudenaert/pablomer/logs/job.%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH -t 03:00:00
#SBATCH -p mi2101x

set -eo pipefail
source ~/.bashrc
conda activate torchenv
ulimit -n 65536 || true
export TORCH_BLAS_PREFER_HIPBLASLT=0
export MIOPEN_USER_DB_PATH="/tmp/miopen_${SLURM_JOB_ID}"
export PYTHONPATH=/work1/jeroenaudenaert/pablomer/tess-generative:${PYTHONPATH:-}

DISCORD="${DISCORD:-https://discord.com/api/webhooks/1499879903464788150/EiosG8MxzpVp4aGsb9FkxV7J-xEpZMABK4yTiwSBdEZ5GA7onfIKvBtHJOaPGli8SGv5}"
CKPT="${CKPT:-/work1/jeroenaudenaert/pablomer/tess-generative/galaxy_images/galaxy_model/checkpoints/hier-small/snapshot.ckpt}"
LABEL="${LABEL:-hier_small}"
NEIGHBORS_H5="${NEIGHBORS_H5:-/work1/jeroenaudenaert/pablomer/data/neighbours_v2.h5}"

GALAXY_MODEL_DIR="/work1/jeroenaudenaert/pablomer/tess-generative/galaxy_images/galaxy_model"
GALLERY_DIR="${GALAXY_MODEL_DIR}/neighbor_search_lenses/galleries/${LABEL}"
mkdir -p "$GALLERY_DIR"

notify() {
    curl -s -X POST "$DISCORD" -H "Content-Type: application/json" \
        -d "{\"content\": \"$1\"}" > /dev/null || true
}

cd "$GALAXY_MODEL_DIR"

for MODE in spatial_flat global_vec; do
    GALLERY="${GALLERY_DIR}/gallery_${MODE}.h5"
    OUT_DIR="${GALAXY_MODEL_DIR}/neighbor_search_lenses/outputs_${LABEL}_${MODE}"

    if [ ! -f "$GALLERY" ]; then
        notify "🔭 [job ${SLURM_JOB_ID}] [${LABEL}/${MODE}] building gallery latents (~110k)..."
        python -m galaxy_images.galaxy_model.neighbor_search_lenses.make_latents_hier \
            --checkpoint "$CKPT" \
            --latent-mode "$MODE" \
            --output "$GALLERY" \
            --neighbors-h5 "$NEIGHBORS_H5" \
            --max-examples 110000 \
            --batch-size 128
        if [ $? -ne 0 ]; then
            notify "❌ [job ${SLURM_JOB_ID}] [${LABEL}/${MODE}] gallery build FAILED"
            exit 1
        fi
        notify "✅ [job ${SLURM_JOB_ID}] [${LABEL}/${MODE}] gallery ready: $GALLERY"
    else
        echo "Gallery exists, skipping build: $GALLERY"
    fi

    notify "🔭 [job ${SLURM_JOB_ID}] [${LABEL}/${MODE}] running NN search on 21 lenses..."
    python neighbor_search_lenses/search_lens_neighbors.py \
        --latents "$GALLERY" \
        --checkpoint "$CKPT" \
        --latent-mode "$MODE" \
        --neighbors-h5 "$NEIGHBORS_H5" \
        --out-dir "$OUT_DIR"

    if [ $? -ne 0 ]; then
        notify "❌ [job ${SLURM_JOB_ID}] [${LABEL}/${MODE}] search FAILED"
        exit 1
    fi

    notify "✅ [job ${SLURM_JOB_ID}] [${LABEL}/${MODE}] search done — sending figures"
    for f in "${OUT_DIR}"/lens_*.png; do
        [ -f "$f" ] || continue
        fname=$(basename "$f")
        curl -s -X POST "$DISCORD" \
            -F "payload_json={\"content\": \"**[${LABEL}/${MODE}] ${fname}**\"}" \
            -F "file=@${f}" > /dev/null || true
    done
done

notify "📬 [job ${SLURM_JOB_ID}] [${LABEL}] all modes done"
