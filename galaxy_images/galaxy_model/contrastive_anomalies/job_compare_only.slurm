#!/bin/bash
#SBATCH -J cmp-top12
#SBATCH -o /home/pablomer/orcd/pool/tess-generative/galaxy_images/galaxy_model/contrastive_anomalies/logs/cmp.%j.out
#SBATCH -e /home/pablomer/orcd/pool/tess-generative/galaxy_images/galaxy_model/contrastive_anomalies/logs/cmp.%j.err
#SBATCH -p mit_normal
#SBATCH -t 01:00:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=64G

# Figures + overlap report. No GPU: this only reads score HDF5s, pulls ~72 stamps
# off the hsc_downstream memmap, and renders matplotlib panels.

set -o pipefail

source /orcd/home/002/pablomer/miniforge3/etc/profile.d/conda.sh
conda activate torchenv

DISCORD="https://discord.com/api/webhooks/1500242707522519183/NqqAxuo5n0Ex4X81vMcq7gDOjrP3oAURAVZtgrZT5CramGNfBGPxuEvC2NnxwYBkfs9j"
WORKDIR="/orcd/pool/007/pablomer/tess-generative/galaxy_images/galaxy_model"
POOL="${POOL:-367k}"
CMP="/orcd/pool/007/pablomer/outputs/contrastive_anomalies/compare474k"

notify() {
    curl -s -X POST "$DISCORD" -H 'Content-Type: application/json' \
        -d "{\"content\": \"$1\"}" > /dev/null
}
notify_file() {
    [ -f "$2" ] && curl -s -X POST "$DISCORD" \
        -F "payload_json={\"content\": \"$1\"}" -F "file=@$2" > /dev/null
}

cd "$WORKDIR"
python -u contrastive_anomalies/compare_474k_top12.py --pool "$POOL" --variants std raw
if [ $? -ne 0 ]; then notify "❌ [job $SLURM_JOB_ID] comparison FAILED"; exit 1; fi

for V in std raw; do
    LBL=$([ "$V" = "std" ] && echo "STANDARDIZED" || echo "UNSTANDARDIZED")
    notify_file "📊 [$LBL / $POOL] all encoders side by side — Base | Contrastive | FM control" "$CMP/joint_all_${POOL}_${V}.png"
    notify_file "📊 [$LBL / $POOL] Base (paper ckpt) physics | instrument" "$CMP/joint_base_${POOL}_${V}.png"
    notify_file "📊 [$LBL / $POOL] Contrastive physics | instrument" "$CMP/joint_contra_${POOL}_${V}.png"
    notify_file "📊 [$LBL / $POOL] FM control physics | instrument" "$CMP/joint_fmctrl_${POOL}_${V}.png"
    notify_file "📄 [$LBL / $POOL] top-12 overlap report" "$CMP/report_${POOL}_${V}.md"
done
notify "🎉 [job $SLURM_JOB_ID] ${POOL} top-12 comparison COMPLETE (366,706 galaxies, flow only, no AION)"
