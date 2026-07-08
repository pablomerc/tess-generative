#!/bin/bash
#SBATCH -J insanom-encode
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
CKPT="${CKPT:-checkpoints/base/snapshot.ckpt}"
EXPDIR=/work1/jeroenaudenaert/pablomer/tess-generative/galaxy_images/galaxy_model/anomaly_detection/redshift_experiment
OUTDIR="$EXPDIR/outputs_instrument"
LATENTS="$OUTDIR/latents_redshift_instrument.h5"
mkdir -p "$OUTDIR"

notify() { curl -s -X POST "$DISCORD" -H "Content-Type: application/json" -d "{\"content\": \"$1\"}" > /dev/null; }

cd /work1/jeroenaudenaert/pablomer/tess-generative/galaxy_images/galaxy_model

notify "🚀 [job $SLURM_JOB_ID] INSTRUMENT-anomaly ENCODE starting (HSC desi_z subset → encoder_2 instrument latent)"
python anomaly_detection/redshift_experiment/encode_redshift_subset.py \
    --checkpoint "$CKPT" --batch-size 256 --encoder instrument --out "$LATENTS"
if [ $? -ne 0 ]; then notify "❌ [job $SLURM_JOB_ID] instrument encode FAILED"; exit 1; fi
notify "✅ [job $SLURM_JOB_ID] instrument encode done → outputs_instrument/latents_redshift_instrument.h5"
