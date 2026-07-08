#!/bin/bash
#SBATCH -J rsanom-cond
#SBATCH -o /work1/jeroenaudenaert/pablomer/logs/job.%j.out
#SBATCH -e /work1/jeroenaudenaert/pablomer/logs/job.%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH -t 04:00:00
#SBATCH -p mi2101x

source ~/.bashrc
conda activate torchenv
export TORCH_BLAS_PREFER_HIPBLASLT=0

DISCORD="https://discord.com/api/webhooks/1496321484338106519/HdI24VGIwsk9IEYoz9MdwMUSmwJ76hJhgIp-TviwYt8Pbnme59KE1xsrHJTM9x3M5eOM"
notify() { curl -s -X POST "$DISCORD" -H "Content-Type: application/json" -d "{\"content\": \"$1\"}" > /dev/null; }

cd /work1/jeroenaudenaert/pablomer/tess-generative/galaxy_images/galaxy_model

notify "🅱️ [job $SLURM_JOB_ID] Job B (conditional NSF p(latent|z)) starting"
python anomaly_detection/redshift_experiment/fit_score_plot_cond.py \
    --top-n 32 --webhook "$DISCORD"   # script default: profile=wide, ctx=z_z2, epochs=100
if [ $? -ne 0 ]; then notify "❌ [job $SLURM_JOB_ID] Job B FAILED"; exit 1; fi
notify "🎉 [job $SLURM_JOB_ID] Job B done"
