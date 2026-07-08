#!/bin/bash
# Submit the v2 redshift-anomaly experiment:
#   encode → Jobs A & B & diagnostic (all three in parallel, depend afterok on encode)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ENC=$(sbatch --parsable "$HERE/job_encode.sh")
echo "encode  job: $ENC"

A=$(sbatch --parsable --dependency=afterok:"$ENC" "$HERE/job_uncond.sh")
echo "Job A   job: $A  (afterok:$ENC)"

B=$(sbatch --parsable --dependency=afterok:"$ENC" "$HERE/job_cond.sh")
echo "Job B   job: $B  (afterok:$ENC)"

# Diagnostic depends on BOTH Job A and Job B because it needs both scores_uncond.npy and scores_cond.npy.
D=$(sbatch --parsable --dependency=afterok:"$A":"$B" "$HERE/job_diag.sh")
echo "Diag    job: $D  (afterok:$A:$B)"

echo
echo "Watch:  squeue -u \$USER"
echo "Logs:   /work1/jeroenaudenaert/pablomer/logs/job.<jobid>.{out,err}"
