#!/bin/bash
#SBATCH --job-name=base-15NB-RND
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --partition=mi2104x
#SBATCH --exclude=k002-005
#SBATCH --output=/work1/jeroenaudenaert/pablomer/logs/base-15NB-RND-DWNVAL_%j.out
#SBATCH --error=/work1/jeroenaudenaert/pablomer/logs/base-15NB-RND-DWNVAL_%j.err
#SBATCH --open-mode=append

export JOB_NAME=base-15NB-RND-DWNVAL
export CONFIG_PATH=/work1/jeroenaudenaert/pablomer/tess-generative/galaxy_images/galaxy_model/configs/neighbors_efficient_15nb_rnd_dwnval.json

source /work1/jeroenaudenaert/pablomer/tess-generative/galaxy_images/galaxy_model/slurm_jobs/_train_template.slurm
