#!/bin/bash
#SBATCH -J counterfactual-anomaly
#SBATCH -o /work1/jeroenaudenaert/pablomer/logs/job.%j.out
#SBATCH -e /work1/jeroenaudenaert/pablomer/logs/job.%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH -t 02:00:00
#SBATCH -p mi2101x

source ~/.bashrc
conda activate torchenv
export TORCH_BLAS_PREFER_HIPBLASLT=0

DISCORD="https://discord.com/api/webhooks/1496321484338106519/HdI24VGIwsk9IEYoz9MdwMUSmwJ76hJhgIp-TviwYt8Pbnme59KE1xsrHJTM9x3M5eOM"

notify() {
    curl -s -X POST "$DISCORD" \
        -H "Content-Type: application/json" \
        -d "{\"content\": \"$1\"}" > /dev/null
}

notify_file() {
    local MSG="$1"
    local FILE="$2"
    if [ -f "$FILE" ]; then
        curl -s -X POST "$DISCORD" \
            -F "content=$MSG" \
            -F "file=@$FILE" > /dev/null
    else
        notify "$MSG (file not found: $FILE)"
    fi
}

cd /work1/jeroenaudenaert/pablomer/tess-generative/galaxy_images/galaxy_model

notify "🔧 [job $SLURM_JOB_ID] counterfactual anomaly experiment starting..."

python anomaly_detection/counterfactual_anomaly_experiment/run_counterfactual.py

if [ $? -ne 0 ]; then
    notify "❌ [job $SLURM_JOB_ID] run_counterfactual FAILED"
    exit 1
fi

OUTBASE="anomaly_detection/counterfactual_anomaly_experiment/outputs"

notify_file "🔀 [counterfactual] A — Style injection (HSC phys)" \
    "${OUTBASE}/exp_A_style_injection/style_injection_grid.png"

notify_file "🔄 [counterfactual] B — Normal reconstruction" \
    "${OUTBASE}/exp_B_normal_recon/normal_recon.png"

notify_file "🧹 [counterfactual] C — Artifact correction (physics pairs)" \
    "${OUTBASE}/exp_C_artifact_correction/artifact_correction.png"

notify_file "🌿 [counterfactual] D — Style injection (Legacy phys)" \
    "${OUTBASE}/exp_D_style_injection_legacy/style_injection_legacy_grid.png"

notify_file "🎲 [counterfactual] E — Artifact correction (random pairs)" \
    "${OUTBASE}/exp_E_artifact_correction_random/artifact_correction_random.png"

notify "🎉 [job $SLURM_JOB_ID] counterfactual experiment ALL DONE!"
