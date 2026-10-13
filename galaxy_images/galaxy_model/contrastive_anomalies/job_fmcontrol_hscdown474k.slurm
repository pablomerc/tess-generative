#!/bin/bash
#SBATCH -J fmctrl-474k
#SBATCH -o /home/pablomer/orcd/pool/tess-generative/galaxy_images/galaxy_model/contrastive_anomalies/logs/fmctrl474k.%j.out
#SBATCH -e /home/pablomer/orcd/pool/tess-generative/galaxy_images/galaxy_model/contrastive_anomalies/logs/fmctrl474k.%j.err
#SBATCH -p mit_normal_gpu
#SBATCH -t 06:00:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH --gres=gpu:1
#SBATCH --mem=128G

# Encode the 474,954-row HSC-downstream pool with the diffusion-ablation FM-control
# checkpoint (encoder_1 = physics, encoder_2 = instrument). Same architecture as
# base/snapshot.ckpt (cross_attention_dim=16, encoder_1/encoder_2), so the existing
# hscdown encoders work unchanged. Flows are fit separately by refit_flows_474k.py
# so that standardized and unstandardized arms share one code path.

set -o pipefail

source /orcd/home/002/pablomer/miniforge3/etc/profile.d/conda.sh
conda activate torchenv
echo "[setup] python=$(which python)  host=$(hostname -s)"

DISCORD="https://discord.com/api/webhooks/1500242707522519183/NqqAxuo5n0Ex4X81vMcq7gDOjrP3oAURAVZtgrZT5CramGNfBGPxuEvC2NnxwYBkfs9j"
WORKDIR="/orcd/pool/007/pablomer/tess-generative/galaxy_images/galaxy_model"
CKPT="/orcd/pool/007/pablomer/checkpoints_new/diffusion_ablation/frozen/fm-control-75k-fixed.ckpt"
SUFFIX="fmctrl_474k"

notify() {
    curl -s -X POST "$DISCORD" -H 'Content-Type: application/json' \
        -d "{\"content\": \"$1\"}" > /dev/null
}

cd "$WORKDIR"
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
notify "🚀 [job $SLURM_JOB_ID] diffusion-ablation **FM control** encode on 474,954 HSC-downstream | host=$(hostname -s) | GPU: ${GPU}"

python -u anomaly_detection/engaging/encode_ours_hscdown.py \
    --checkpoint "$CKPT" --suffix "$SUFFIX" --batch-size 256
if [ $? -ne 0 ]; then notify "❌ [job $SLURM_JOB_ID] FM-control physics encode FAILED"; exit 1; fi
notify "✅ [job $SLURM_JOB_ID] FM-control physics latents done"

python -u anomaly_detection/engaging/encode_instrument_hscdown.py \
    --checkpoint "$CKPT" --suffix "$SUFFIX" --batch-size 256
if [ $? -ne 0 ]; then notify "❌ [job $SLURM_JOB_ID] FM-control instrument encode FAILED"; exit 1; fi
notify "✅ [job $SLURM_JOB_ID] FM-control instrument latents done → refit_flows_474k.py next"
