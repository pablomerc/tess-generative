#!/bin/bash
#SBATCH -J refit-474k
#SBATCH -o /home/pablomer/orcd/pool/tess-generative/galaxy_images/galaxy_model/contrastive_anomalies/logs/refit474k.%j.out
#SBATCH -e /home/pablomer/orcd/pool/tess-generative/galaxy_images/galaxy_model/contrastive_anomalies/logs/refit474k.%j.err
#SBATCH -p mit_normal_gpu
#SBATCH -t 06:00:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH --mem=96G

# Fit every 474k arm through one NSF code path, standardized AND unstandardized.
# Arms are ordered so the ones the comparison actually needs land first; AION is
# 768-D and by far the slowest, so it goes last. Each (arm, variant) is written as
# soon as it finishes and re-runs skip existing files, so a timeout loses nothing.

set -o pipefail

source /orcd/home/002/pablomer/miniforge3/etc/profile.d/conda.sh
conda activate torchenv
echo "[setup] python=$(which python)  host=$(hostname -s)"

DISCORD="https://discord.com/api/webhooks/1500242707522519183/NqqAxuo5n0Ex4X81vMcq7gDOjrP3oAURAVZtgrZT5CramGNfBGPxuEvC2NnxwYBkfs9j"
WORKDIR="/orcd/pool/007/pablomer/tess-generative/galaxy_images/galaxy_model"

notify() {
    curl -s -X POST "$DISCORD" -H 'Content-Type: application/json' \
        -d "{\"content\": \"$1\"}" > /dev/null
}
notify_file() {
    [ -f "$2" ] && curl -s -X POST "$DISCORD" \
        -F "payload_json={\"content\": \"$1\"}" -F "file=@$2" > /dev/null
}

cd "$WORKDIR"
notify "🧮 [job $SLURM_JOB_ID] refitting all 474k NSF flows (7 arms x std/raw) | host=$(hostname -s)"

POOL="${POOL:-367k}"

# No AION arm — dropped on request. 6 arms x {std,raw} = 12 NSF fits.
#
# Each fit is tiny for the GPU (293k x 64 float32 = 75 MB of data, hidden [64,64]),
# so running them one after another leaves the card mostly idle. Fan out over three
# processes sharing the one GPU the QOS allows — each owns a disjoint pair of arms,
# so their output files never collide. ~3x less wall time than sequential.
LOGD="contrastive_anomalies/logs"
run_arms() {
    python -u contrastive_anomalies/refit_flows_474k.py \
        --pool "$POOL" --variants std raw --arms "$@"
}
run_arms base-physics   base-instrument   > "${LOGD}/refit_base.${SLURM_JOB_ID}.log"   2>&1 &
P1=$!
run_arms contra-physics contra-instrument > "${LOGD}/refit_contra.${SLURM_JOB_ID}.log" 2>&1 &
P2=$!
run_arms fmctrl-physics fmctrl-instrument > "${LOGD}/refit_fmctrl.${SLURM_JOB_ID}.log" 2>&1 &
P3=$!

RC=0
for P in $P1 $P2 $P3; do wait $P || RC=1; done
if [ $RC -ne 0 ]; then
    notify "❌ [job $SLURM_JOB_ID] refit FAILED — see refit_{base,contra,fmctrl}.${SLURM_JOB_ID}.log"
    tail -20 "${LOGD}"/refit_*."${SLURM_JOB_ID}".log
    exit 1
fi
notify "✅ [job $SLURM_JOB_ID] all 12 ${POOL} flows refit — building comparison"

python -u contrastive_anomalies/compare_474k_top12.py --pool "$POOL" --variants std raw
if [ $? -ne 0 ]; then notify "❌ [job $SLURM_JOB_ID] comparison FAILED"; exit 1; fi

CMP="/orcd/pool/007/pablomer/outputs/contrastive_anomalies/compare474k"
for V in std raw; do
    LBL=$([ "$V" = "std" ] && echo "STANDARDIZED" || echo "UNSTANDARDIZED")
    notify_file "📊 [$LBL / $POOL] Base (paper ckpt) physics | instrument" "$CMP/joint_base_${POOL}_${V}.png"
    notify_file "📊 [$LBL / $POOL] Contrastive physics | instrument" "$CMP/joint_contra_${POOL}_${V}.png"
    notify_file "📊 [$LBL / $POOL] FM control physics | instrument" "$CMP/joint_fmctrl_${POOL}_${V}.png"
    notify_file "📊 [$LBL / $POOL] all encoders side by side" "$CMP/joint_all_${POOL}_${V}.png"
    notify_file "📄 [$LBL / $POOL] top-12 overlap report" "$CMP/report_${POOL}_${V}.md"
done
notify "🎉 [job $SLURM_JOB_ID] ${POOL} top-12 comparison COMPLETE"
