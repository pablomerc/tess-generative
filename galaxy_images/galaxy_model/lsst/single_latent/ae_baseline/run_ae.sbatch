#!/bin/bash
#SBATCH --job-name=lsst_ae
#SBATCH --partition=mi2101x
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --time=12:00:00
#SBATCH --output=/work1/jeroenaudenaert/pablomer/logs/lsst_ae-%j.log

# LSST AE baseline: train + latent UMAPs to WandB + anomaly stage + Discord.
export TORCH_BLAS_PREFER_HIPBLASLT=0                 # MI210 hipBLASLt bug
export DISCORD_WEBHOOK_URL=$(cat ~/.config/discord_webhook)

source /work1/jeroenaudenaert/pablomer/miniconda3/etc/profile.d/conda.sh
conda activate torchenv

cd /work1/jeroenaudenaert/pablomer/tess-generative
export PYTHONPATH=.
python -m galaxy_images.galaxy_model.lsst.single_latent.ae_baseline.train "$@"
