#!/bin/bash
# Submit the full pipeline (all 474k galaxies) to SLURM.
# Stage 2 is held until Stage 1 finishes successfully.
# Edit CHECKPOINT and SUFFIX below, then: bash submit_full_run.sh

CHECKPOINT="/work1/jeroenaudenaert/pablomer/tess-generative/galaxy_images/galaxy_model/outputs/neighbors_all_attn/2026-04-05_2/checkpoints/latest-step=step=104000.ckpt"
SUFFIX="neighbors_all_attn_104k"

# -----------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Patch the SLURM scripts with the chosen checkpoint and suffix
sed -i \
    "s|CHECKPOINT=\".*\"|CHECKPOINT=\"${CHECKPOINT}\"|" \
    "${SCRIPT_DIR}/run_prepare_hsc.slurm"

sed -i \
    "s|SUFFIX=\".*\"|SUFFIX=\"${SUFFIX}\"|g" \
    "${SCRIPT_DIR}/run_prepare_hsc.slurm" \
    "${SCRIPT_DIR}/run_predict_hsc.slurm"

# Submit Stage 1
JID=$(sbatch --parsable "${SCRIPT_DIR}/run_prepare_hsc.slurm")
echo "Submitted Stage 1 (prepare):  job ${JID}"

# Submit Stage 2+3 with dependency on Stage 1
JID2=$(sbatch --parsable --dependency=afterok:${JID} "${SCRIPT_DIR}/run_predict_hsc.slurm")
echo "Submitted Stage 2+3 (predict + compare): job ${JID2} (held until ${JID} finishes)"

echo ""
echo "Monitor with:"
echo "  squeue -u \$USER"
echo "  tail -f /work1/jeroenaudenaert/pablomer/logs/hsc_prepare_${JID}.out"
