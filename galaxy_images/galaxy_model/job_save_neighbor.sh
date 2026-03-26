#!/bin/bash

#SBATCH -J save_neighbor          # Job name
#SBATCH -o /work1/jeroenaudenaert/pablomer/logs/job.%j.out   # stdout log (%j expands to jobId)
#SBATCH -e /work1/jeroenaudenaert/pablomer/logs/job.%j.err   # stderr log
#SBATCH -N 1                      # 1 node
#SBATCH -n 1                      # 1 task
#SBATCH -c 10                     # CPUs for NUM_WORKERS=8 + overhead
#SBATCH -t 12:00:00               # 12-hour wall time
#SBATCH -p mi2101x                # 0.1x charge multiplier; no GPU compute needed

# --- Environment ---
source ~/.bashrc
conda activate torchenv   # replace with your env name (run `conda env list` to check)

# --- Run ---
cd /work1/jeroenaudenaert/pablomer/tess-generative

python galaxy_images/galaxy_model/save_neighbor_gemini.py
