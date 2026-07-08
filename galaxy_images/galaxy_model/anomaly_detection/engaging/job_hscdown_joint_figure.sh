#!/bin/bash
#SBATCH -J anom-hscdown-joint
#SBATCH -o /home/pablomer/orcd/pool/tess-generative/galaxy_images/galaxy_model/anomaly_detection/engaging/logs/joint.%j.out
#SBATCH -e /home/pablomer/orcd/pool/tess-generative/galaxy_images/galaxy_model/anomaly_detection/engaging/logs/joint.%j.err
#SBATCH -p mit_normal_gpu
#SBATCH -t 00:30:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=32G

set -o pipefail

source /orcd/home/002/pablomer/miniforge3/etc/profile.d/conda.sh
conda activate torchenv
echo "[setup] python=$(which python)"

DISCORD="https://discord.com/api/webhooks/1501287660579192892/Nijc62HtCBt28dabGDsXskmIVlXsrz6OofxDn3tkoRFrX4bNqx0MGSRllTx2RXhWurpB"
SUFFIX="hscdown_474k"
WORKDIR="/orcd/pool/007/pablomer/tess-generative/galaxy_images/galaxy_model"
FIGDIR="${WORKDIR}/anomaly_detection/outputs/figures_${SUFFIX}"

notify() {
    curl -s -X POST "$DISCORD" \
        -H "Content-Type: application/json" \
        -d "{\"content\": \"$1\"}" > /dev/null
}
notify_file() {
    if [ -f "$2" ]; then
        curl -s -X POST "$DISCORD" \
            -F "payload_json={\"content\": \"$1\"}" \
            -F "file=@$2" > /dev/null
    else
        notify "$1 (file not found: $2)"
    fi
}

cd "$WORKDIR"

notify "🧩 [job $SLURM_JOB_ID] joint figure (3 columns × top-12) — building..."
python anomaly_detection/engaging/build_joint_figure_hscdown.py \
    --suffix "$SUFFIX" \
    --out-dir "$FIGDIR"
if [ $? -ne 0 ]; then notify "❌ [job $SLURM_JOB_ID] joint figure FAILED"; exit 1; fi

for METHOD in flow iforest; do
    OUT="${FIGDIR}/joint_top12_${METHOD}_${SUFFIX}.png"
    notify_file "🧩 [hscdown JOINT ${METHOD}] 3 encoders × top-12" "$OUT"
done

notify "🎉 [job $SLURM_JOB_ID] joint figure DONE"
