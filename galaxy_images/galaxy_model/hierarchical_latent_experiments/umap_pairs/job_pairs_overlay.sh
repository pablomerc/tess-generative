#!/bin/bash
#SBATCH -J umap-pairs-overlay
#SBATCH -o /work1/jeroenaudenaert/pablomer/logs/job.%j.out
#SBATCH -e /work1/jeroenaudenaert/pablomer/logs/job.%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH -t 00:15:00
#SBATCH -p mi2101x

set -eo pipefail
source ~/.bashrc
conda activate torchenv
ulimit -n 65536 || true
export PYTHONPATH=/work1/jeroenaudenaert/pablomer/tess-generative:${PYTHONPATH:-}

DATA_DIR="${DATA_DIR:-/work1/jeroenaudenaert/pablomer/tess-generative/galaxy_images/galaxy_model/hierarchical_latent_experiments/umap_pairs/outputs/spatial_flat__hier_small}"
VARIANT="${VARIANT:-spatial_flat}"
K="${K:-8}"
SEED="${SEED:-0}"

cd /work1/jeroenaudenaert/pablomer/tess-generative
python -m galaxy_images.galaxy_model.hierarchical_latent_experiments.umap_pairs.plot_pairs_overlay \
    --data-dir "$DATA_DIR" --variant "$VARIANT" --k "$K" --seed "$SEED"
