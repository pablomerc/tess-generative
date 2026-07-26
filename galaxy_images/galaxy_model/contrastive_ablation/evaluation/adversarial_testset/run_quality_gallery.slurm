#!/bin/bash
#SBATCH --job-name=quality-gallery
#SBATCH --partition=mit_normal
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/orcd/pool/007/pablomer/tess-generative/galaxy_images/galaxy_model/contrastive_ablation/logs/%x-%j.out

set -eo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENGAGING="${ENGAGING:-$HERE/../../../downstream_evaluation/engaging}"
OVERLAP="${OVERLAP:-$ENGAGING/outputs/index/overlap.parquet}"
HSC_DIR="${HSC_DIR:-/home/pablomer/orcd/scratch/hsc_downstream}"
LEGACY_DIR="${LEGACY_DIR:-/home/pablomer/orcd/scratch/legacy_downstream_full/full_1M}"
OUT_DIR="${OUT_DIR:-$HERE/outputs}"

source ~/.bashrc
conda activate "${CONDA_ENV:-torchenv}"
set -u
PYTHON="${PYTHON:-python}"

mkdir -p "$OUT_DIR" "$OUT_DIR/gallery"

"$PYTHON" "$HERE/quality_table.py" audit \
  --overlap "$OVERLAP" \
  --hsc-catalog "$HSC_DIR/catalog.parquet" \
  --legacy-catalog "$LEGACY_DIR/catalog.parquet" \
  --report "$OUT_DIR/schema_audit.json"

"$PYTHON" "$HERE/quality_table.py" build \
  --overlap "$OVERLAP" \
  --hsc-catalog "$HSC_DIR/catalog.parquet" \
  --legacy-catalog "$LEGACY_DIR/catalog.parquet" \
  --out "$OUT_DIR/quality_table.parquet" \
  --candidates "$OUT_DIR/quality_candidates_top200.csv" \
  --top-n "${TOP_N:-200}" \
  --moments-unit "${MOMENTS_UNIT:-pixel2}" \
  --hsc-pixel-scale "${HSC_PIXEL_SCALE:-0.168}"

"$PYTHON" "$HERE/make_gallery.py" \
  --quality-table "$OUT_DIR/quality_table.parquet" \
  --hsc-bin "$HSC_DIR/hsc_flux.bin" \
  --legacy-bin "$LEGACY_DIR/legacy_flux.bin" \
  --out-dir "$OUT_DIR/gallery" \
  --n-per-stratum "${GALLERY_N:-12}" \
  --seed "${SEED:-0}"

echo "Quality table and galleries complete. Inspect them before freezing a manifest."
