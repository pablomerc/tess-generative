#!/bin/bash
#SBATCH -J anom-hscdown-aion
#SBATCH -o /home/pablomer/orcd/pool/tess-generative/galaxy_images/galaxy_model/anomaly_detection/engaging/logs/aion.%j.out
#SBATCH -e /home/pablomer/orcd/pool/tess-generative/galaxy_images/galaxy_model/anomaly_detection/engaging/logs/aion.%j.err
#SBATCH -p mit_normal_gpu
#SBATCH -t 06:00:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=192G

set -o pipefail

source /orcd/home/002/pablomer/miniforge3/etc/profile.d/conda.sh
conda activate torchenv
echo "[setup] python=$(which python)"

DISCORD="https://discord.com/api/webhooks/1501287660579192892/Nijc62HtCBt28dabGDsXskmIVlXsrz6OofxDn3tkoRFrX4bNqx0MGSRllTx2RXhWurpB"
SUFFIX="hscdown_474k"
WORKDIR="/orcd/pool/007/pablomer/tess-generative/galaxy_images/galaxy_model"
OUTDIR="${WORKDIR}/anomaly_detection/outputs"
FIGDIR="${OUTDIR}/figures_${SUFFIX}"
LATENTS="${OUTDIR}/anomaly_latents_aion_${SUFFIX}.h5"
SCORES="${OUTDIR}/anomaly_scores_aion_${SUFFIX}.h5"

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

GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1)
notify "🚀 [job $SLURM_JOB_ID] hscdown-AION start | host=$(hostname -s) | GPU: ${GPU_INFO}"

# ── Step 1: Encode (AION) ──────────────────────────────────────────────────────
notify "🔢 [job $SLURM_JOB_ID] (1/3) encode_aion_hscdown.py — 474,954 examples"
python anomaly_detection/engaging/encode_aion_hscdown.py \
    --suffix "$SUFFIX" \
    --batch-size 64
if [ $? -ne 0 ]; then notify "❌ [job $SLURM_JOB_ID] encode FAILED"; exit 1; fi
notify "✅ [job $SLURM_JOB_ID] (1/3) latents → $(basename $LATENTS)"

# ── Step 2: Fit (IForest + NSF) — raw 768-d, NO PCA ───────────────────────────
notify "🧮 [job $SLURM_JOB_ID] (2/3) fit IForest + NSF on raw 768-d AION embeddings (no PCA)"
python anomaly_detection/engaging/fit_scores_hscdown.py \
    --encoder aion \
    --suffix "$SUFFIX"
if [ $? -ne 0 ]; then notify "❌ [job $SLURM_JOB_ID] fit FAILED"; exit 1; fi
notify "✅ [job $SLURM_JOB_ID] (2/3) scores → $(basename $SCORES)"

# ── Step 3: Top-12 plots (flow + iforest) ─────────────────────────────────────
notify "🖼️ [job $SLURM_JOB_ID] (3/3) rendering top-12 plots"
mkdir -p "$FIGDIR"
for METHOD in flow iforest; do
    OUT="${FIGDIR}/top12_aion_${METHOD}_${SUFFIX}.png"
    python anomaly_detection/engaging/plot_top12_hscdown.py \
        --scores "$SCORES" \
        --score-key "aion/hsc_mean_full/${METHOD}" \
        --label "AION (no PCA, 768-d) — ${METHOD}" \
        --out "$OUT"
    if [ $? -ne 0 ]; then notify "⚠️ [job $SLURM_JOB_ID] plot ${METHOD} FAILED"; continue; fi
    notify_file "📊 [hscdown AION ${METHOD}] top-12" "$OUT"
done

notify "🎉 [job $SLURM_JOB_ID] hscdown-AION ALL DONE"
