#!/bin/bash
#SBATCH --job-name=build_lsst_eff
#SBATCH --partition=mi3001x
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=03:00:00
#SBATCH --output=/work1/jeroenaudenaert/pablomer/logs/build_lsst_efficient-%j.log

# CPU-only: streams ~70 GB of DP2 cutout shards into data/lsst_efficient/
set -euo pipefail
cd /work1/jeroenaudenaert/pablomer/tess-generative
exec /work1/jeroenaudenaert/pablomer/miniconda3/envs/torchenv/bin/python \
  galaxy_images/galaxy_model/lsst/data_preprocessing/build_lsst_efficient.py
