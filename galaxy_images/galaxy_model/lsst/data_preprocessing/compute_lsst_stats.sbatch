#!/bin/bash
#SBATCH --job-name=lsst_stats
#SBATCH --partition=mi3001x
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=/work1/jeroenaudenaert/pablomer/logs/lsst_stats-%j.log

# CPU-only: derives LSST clamp/norm constants from data/lsst_efficient/
cd /work1/jeroenaudenaert/pablomer/tess-generative
export PYTHONPATH=.
/work1/jeroenaudenaert/pablomer/miniconda3/envs/torchenv/bin/python \
  galaxy_images/galaxy_model/lsst/data_preprocessing/compute_lsst_stats.py
