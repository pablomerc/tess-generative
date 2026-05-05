#!/bin/bash
#SBATCH --job-name=base6x6-15NB
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --partition=mi2104x
#SBATCH --exclude=k002-005,k003-007
#SBATCH --output=/work1/jeroenaudenaert/pablomer/logs/base6x6-15NB_%j.out
#SBATCH --error=/work1/jeroenaudenaert/pablomer/logs/base6x6-15NB_%j.err
#SBATCH --open-mode=append

export JOB_NAME=base6x6-15NB
export CONFIG_PATH=/work1/jeroenaudenaert/pablomer/tess-generative/galaxy_images/galaxy_model/configs/neighbors_efficient_base6x6_15nb.json

source /work1/jeroenaudenaert/pablomer/tess-generative/galaxy_images/galaxy_model/slurm_jobs/_train_template.slurm
