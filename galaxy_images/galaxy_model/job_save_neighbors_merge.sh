#!/bin/bash

#SBATCH -J save_neighbors_merge   # Job name
#SBATCH -o /work1/jeroenaudenaert/pablomer/logs/job.%j.out   # stdout log
#SBATCH -e /work1/jeroenaudenaert/pablomer/logs/job.%j.err   # stderr log
#SBATCH -N 1                      # 1 node
#SBATCH -n 1                      # 1 task
#SBATCH -c 2                      # CPUs (pure IO, no parallelism needed)
#SBATCH -t 00:30:00               # 30 min should be plenty
#SBATCH -p devel                  # devel partition; cheap and fast for IO-only jobs

# --- Environment ---
source ~/.bashrc
conda activate torchenv

# --- Run ---
cd /work1/jeroenaudenaert/pablomer/tess-generative

python galaxy_images/galaxy_model/save_neighbors_gemini_merge.py
