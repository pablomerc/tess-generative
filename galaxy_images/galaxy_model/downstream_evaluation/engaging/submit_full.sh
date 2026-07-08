#!/bin/bash
# Submit the full engaging downstream evaluation pipeline.
#
# Job graph:
#   build  ──┬──► prepare ──► predict
#            └──► overlap-viz (discord)
#
# Each sbatch is queued immediately; the dependency lines tell SLURM to wait.

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

JID_BUILD=$(sbatch --parsable run_build.slurm)
echo "build:           ${JID_BUILD}"

JID_PREP=$(sbatch --parsable --dependency=afterok:${JID_BUILD} run_prepare.slurm)
echo "prepare:         ${JID_PREP}    (after build)"

JID_PRED=$(sbatch --parsable --dependency=afterok:${JID_PREP} run_predict.slurm)
echo "predict+plot:    ${JID_PRED}    (after prepare)"

JID_VIZ=$(sbatch --parsable --dependency=afterok:${JID_BUILD} run_overlap_check.slurm)
echo "overlap-viz:     ${JID_VIZ}    (after build, parallel to prepare)"

echo ""
echo "queue:"
squeue -u "${USER}"
