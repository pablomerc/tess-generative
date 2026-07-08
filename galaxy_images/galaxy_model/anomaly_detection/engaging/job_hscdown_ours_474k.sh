#!/bin/bash
#SBATCH -J anom-hscdown-ours
#SBATCH -o /home/pablomer/orcd/pool/tess-generative/galaxy_images/galaxy_model/anomaly_detection/engaging/logs/ours.%j.out
#SBATCH -e /home/pablomer/orcd/pool/tess-generative/galaxy_images/galaxy_model/anomaly_detection/engaging/logs/ours.%j.err
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
echo "[setup] python=$(which python)"

DISCORD="https://discord.com/api/webhooks/1501287660579192892/Nijc62HtCBt28dabGDsXskmIVlXsrz6OofxDn3tkoRFrX4bNqx0MGSRllTx2RXhWurpB"
SUFFIX="hscdown_474k"
WORKDIR="/orcd/pool/007/pablomer/tess-generative/galaxy_images/galaxy_model"
OUTDIR="${WORKDIR}/anomaly_detection/outputs"
FIGDIR="${OUTDIR}/figures_${SUFFIX}"
ENG="${WORKDIR}/anomaly_detection/engaging"
LATENTS="${OUTDIR}/anomaly_latents_ours_${SUFFIX}.h5"
SCORES="${OUTDIR}/anomaly_scores_ours_${SUFFIX}.h5"

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
notify "🚀 [job $SLURM_JOB_ID] hscdown-ours-physics start | host=$(hostname -s) | GPU: ${GPU_INFO}"

# ── Step 1: Encode ─────────────────────────────────────────────────────────────
notify "🔢 [job $SLURM_JOB_ID] (1/3) encode_ours_hscdown.py — 474,954 examples"
python anomaly_detection/engaging/encode_ours_hscdown.py \
    --suffix "$SUFFIX" \
    --batch-size 256
if [ $? -ne 0 ]; then notify "❌ [job $SLURM_JOB_ID] encode FAILED"; exit 1; fi
notify "✅ [job $SLURM_JOB_ID] (1/3) latents → $(basename $LATENTS)"

# ── Step 2: Fit (IForest + NSF) ───────────────────────────────────────────────
notify "🧮 [job $SLURM_JOB_ID] (2/3) fit IForest + NSF on hsc_flat"
python anomaly_detection/engaging/fit_scores_hscdown.py \
    --encoder ours-physics \
    --suffix "$SUFFIX"
if [ $? -ne 0 ]; then notify "❌ [job $SLURM_JOB_ID] fit FAILED"; exit 1; fi
notify "✅ [job $SLURM_JOB_ID] (2/3) scores → $(basename $SCORES)"

# ── Step 3: Top-12 plots (flow + iforest) ─────────────────────────────────────
notify "🖼️ [job $SLURM_JOB_ID] (3/3) rendering top-12 plots"
mkdir -p "$FIGDIR"
for METHOD in flow iforest; do
    OUT="${FIGDIR}/top12_ours_${METHOD}_${SUFFIX}.png"
    python anomaly_detection/engaging/plot_top12_hscdown.py \
        --scores "$SCORES" \
        --score-key "ours/hsc_flat/${METHOD}" \
        --label "Ours (Physics) — ${METHOD} — hsc_flat" \
        --out "$OUT"
    if [ $? -ne 0 ]; then notify "⚠️ [job $SLURM_JOB_ID] plot ${METHOD} FAILED"; continue; fi
    notify_file "📊 [hscdown ours-physics ${METHOD}] top-12" "$OUT"
done

notify "🎉 [job $SLURM_JOB_ID] hscdown-ours-physics ALL DONE"
