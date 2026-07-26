#!/bin/bash
#SBATCH --job-name=fig5-repro
#SBATCH --partition=pg_mki_aryeh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=05:00:00
#SBATCH --output=/home/pablomer/orcd/pool/tess-generative/galaxy_images/galaxy_model/scale_ablation/lens_recall/outputs/%x_%j.out
#SBATCH --error=/home/pablomer/orcd/pool/tess-generative/galaxy_images/galaxy_model/scale_ablation/lens_recall/outputs/%x_%j.err
#
# Reproduce the paper's Fig. 5 lens retrieval with the paper checkpoint (base-87k), via BOTH
# image sources, so we learn two things in one job:
#   1. does the retrieval reproduce the recorded neighbour ranks?          (fidelity)
#   2. does the cheap pre-baked RAM48 store give the same answer as the    (feasibility of
#      faithful raw-stamp path?                                             the 10-ckpt sweep)
# It also prints img/s for each source, which is what sizes the sweep.

set -eo pipefail
REPO="/home/pablomer/orcd/pool/tess-generative"
LR="${REPO}/galaxy_images/galaxy_model/scale_ablation/lens_recall"
BASE="/orcd/pool/007/pablomer/checkpoints_new/base/snapshot.ckpt"
mkdir -p "${LR}/outputs" "${LR}/latents"

WEBHOOK="${DISCORD_WEBHOOK:-https://discord.com/api/webhooks/1500242707522519183/NqqAxuo5n0Ex4X81vMcq7gDOjrP3oAURAVZtgrZT5CramGNfBGPxuEvC2NnxwYBkfs9j}"
_ping() { curl -sS -X POST -H 'Content-Type: application/json' -d "{\"content\": \"$1\"}" "${WEBHOOK}" >/dev/null 2>&1 || true; }
_pingf() { curl -sS -F "payload_json={\"content\": \"$1\"}" -F "file=@$2" "${WEBHOOK}" >/dev/null 2>&1 || true; }
_ping "fig5-repro 🔭 start · base-87k · job ${SLURM_JOB_ID:-?}"
trap '_ping "fig5-repro $([ $? -eq 0 ] && echo ✅ || echo ❌) · job ${SLURM_JOB_ID:-?}"' EXIT

source ~/.bashrc
conda activate torchenv
cd "${REPO}/galaxy_images/galaxy_model"
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

for SRC in ram48 h5; do
  echo "############ source=${SRC} ############"
  srun --unbuffered python -u "${LR}/encode_gallery.py" \
    --checkpoint "${BASE}" --source "${SRC}" \
    --out "${LR}/latents/base87k_${SRC}.h5" --batch-size 256 --num-workers 8

  srun --unbuffered python -u "${LR}/reproduce_fig5.py" \
    --gallery "${LR}/latents/base87k_${SRC}.h5" \
    --out-prefix "${LR}/outputs/fig5_base87k_${SRC}" \
    --label "base-87k / source=${SRC}" | tee "${LR}/outputs/fig5_base87k_${SRC}.txt"

  MATCH=$(python -c "import json;d=json.load(open('${LR}/outputs/fig5_base87k_${SRC}.json'))['spaces']['physics'];print(f\"{d['recorded_ranks_matched']}/{d['recorded_ranks_checked']}\")")
  _pingf "**Fig. 5 reproduction — base-87k (source=\`${SRC}\`)** · physics space reproduces **${MATCH}** recorded ranks. Top row = physics, bottom = instrument (disentanglement control). Green titles = rank matches the run that made the paper figure." \
         "${LR}/outputs/fig5_base87k_${SRC}.png"
done

echo "=== throughput summary (drives the 10-checkpoint sweep estimate) ==="
grep -hE "img/s|loaded in|wrote .*\.h5" "${LR}/outputs/fig5-repro_${SLURM_JOB_ID}.out" | tail -20 || true
