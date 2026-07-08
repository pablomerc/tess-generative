#!/bin/bash
# Submit the 3 encoder jobs in parallel and a 4th joint-figure job that waits
# for all 3 to succeed (`--dependency=afterok`).
#
#   $ ./submit_all.sh
#
# Prints all 4 job IDs.

set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "${DIR}/logs"

JID_OURS=$(sbatch --parsable "${DIR}/job_hscdown_ours_474k.sh")
JID_INS=$(sbatch  --parsable "${DIR}/job_hscdown_ins_474k.sh")
JID_AION=$(sbatch --parsable "${DIR}/job_hscdown_aion_474k.sh")
JID_JOINT=$(sbatch --parsable \
    --dependency="afterok:${JID_OURS}:${JID_INS}:${JID_AION}" \
    "${DIR}/job_hscdown_joint_figure.sh")

echo "Submitted jobs:"
echo "  ours-physics    : ${JID_OURS}"
echo "  ours-instrument : ${JID_INS}"
echo "  AION (no PCA)   : ${JID_AION}"
echo "  joint figure    : ${JID_JOINT}  (afterok:${JID_OURS}:${JID_INS}:${JID_AION})"
