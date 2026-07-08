#!/bin/bash
#SBATCH --job-name=base-ins-flat
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --partition=mi2104x
#SBATCH --exclude=k002-005,k003-001,k003-007
#SBATCH --output=/work1/jeroenaudenaert/pablomer/logs/base_ins_flatten_%j.out
#SBATCH --error=/work1/jeroenaudenaert/pablomer/logs/base_ins_flatten_%j.err
#SBATCH --open-mode=append

export JOB_NAME=base_ins_flatten
export CONFIG_PATH=/work1/jeroenaudenaert/pablomer/tess-generative/galaxy_images/galaxy_model/configs/neighbors_efficient_base_ins_flatten.json

source /work1/jeroenaudenaert/pablomer/tess-generative/galaxy_images/galaxy_model/slurm_jobs/_train_template.slurm
