#!/bin/bash
#SBATCH -J contra-474k
#SBATCH -o /home/pablomer/orcd/pool/tess-generative/galaxy_images/galaxy_model/contrastive_anomalies/logs/contra474k.%j.out
#SBATCH -e /home/pablomer/orcd/pool/tess-generative/galaxy_images/galaxy_model/contrastive_anomalies/logs/contra474k.%j.err
#SBATCH -p mit_normal_gpu
#SBATCH -t 06:00:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH --gres=gpu:1
#SBATCH --mem=128G

set -o pipefail

source /orcd/home/002/pablomer/miniforge3/etc/profile.d/conda.sh
conda activate torchenv
echo "[setup] python=$(which python)  host=$(hostname -s)"

DISCORD="https://discord.com/api/webhooks/1500242707522519183/NqqAxuo5n0Ex4X81vMcq7gDOjrP3oAURAVZtgrZT5CramGNfBGPxuEvC2NnxwYBkfs9j"
WORKDIR="/orcd/pool/007/pablomer/tess-generative/galaxy_images/galaxy_model"
CKPT="${CKPT:-/orcd/pool/007/pablomer/checkpoints_new/contrastive-spatial-conv1x1/best-epoch=53-step=83000.ckpt}"
TAG="${TAG:-spatial-conv1x1}"

notify() {
    curl -s -X POST "$DISCORD" -H 'Content-Type: application/json' \
        -d "{\"content\": \"$1\"}" > /dev/null
}

cd "$WORKDIR"
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
notify "🚀 [job $SLURM_JOB_ID] contrastive **${TAG}** on the full 474,954 HSC-downstream pool | host=$(hostname -s) | GPU: ${GPU}"

python -u contrastive_anomalies/encode_score_hscdown474k.py \
    --ckpt "$CKPT" \
    --tag "$TAG" \
    --batch-size 256 \
    --num-workers 12 \
    --reuse-latents
RC=$?

if [ $RC -ne 0 ]; then
    notify "❌ [job $SLURM_JOB_ID] contrastive ${TAG} 474k FAILED (rc=$RC)"
    exit $RC
fi
notify "✅ [job $SLURM_JOB_ID] contrastive ${TAG} 474k done → scores_${TAG}_hscdown_474k.h5"
