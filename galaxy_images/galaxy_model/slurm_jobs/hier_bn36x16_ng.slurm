#!/bin/bash
#SBATCH --job-name=hier-NG
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --partition=mi2104x
#SBATCH --exclude=k002-005
#SBATCH --output=/work1/jeroenaudenaert/pablomer/logs/hier-NG_%j.out
#SBATCH --error=/work1/jeroenaudenaert/pablomer/logs/hier-NG_%j.err
#SBATCH --open-mode=append

export JOB_NAME=hier-NG
export CONFIG_PATH=/work1/jeroenaudenaert/pablomer/tess-generative/galaxy_images/galaxy_model/configs/neighbors_efficient_hier_bn36x16_ng.json

source /work1/jeroenaudenaert/pablomer/tess-generative/galaxy_images/galaxy_model/slurm_jobs/_train_template.slurm
