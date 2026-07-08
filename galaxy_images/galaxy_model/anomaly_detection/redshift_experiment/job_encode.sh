#!/bin/bash
#SBATCH -J rsanom-encode
#SBATCH -o /work1/jeroenaudenaert/pablomer/logs/job.%j.out
#SBATCH -e /work1/jeroenaudenaert/pablomer/logs/job.%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH -t 04:00:00
#SBATCH -p mi2101x

source ~/.bashrc
conda activate torchenv
export TORCH_BLAS_PREFER_HIPBLASLT=0

DISCORD="https://discord.com/api/webhooks/1496321484338106519/HdI24VGIwsk9IEYoz9MdwMUSmwJ76hJhgIp-TviwYt8Pbnme59KE1xsrHJTM9x3M5eOM"
# Default checkpoint: registered baseline. For parity with the prior ours/hsc_mean anomaly
# figure, swap to: outputs/neighbors_all_attn/2026-04-05/checkpoints/best-epoch=228-step=87000.ckpt
CKPT="${CKPT:-checkpoints/base/snapshot.ckpt}"

notify() { curl -s -X POST "$DISCORD" -H "Content-Type: application/json" -d "{\"content\": \"$1\"}" > /dev/null; }

cd /work1/jeroenaudenaert/pablomer/tess-generative/galaxy_images/galaxy_model

notify "🚀 [job $SLURM_JOB_ID] redshift-anomaly ENCODE starting (HSC desi_z subset → physics latent)"
python anomaly_detection/redshift_experiment/encode_redshift_subset.py \
    --checkpoint "$CKPT" \
    --batch-size 256
if [ $? -ne 0 ]; then notify "❌ [job $SLURM_JOB_ID] encode FAILED"; exit 1; fi
notify "✅ [job $SLURM_JOB_ID] encode done → outputs/latents_redshift.h5"
