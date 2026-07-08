#!/bin/bash
# Submit the INSTRUMENT-space redshift-anomaly experiment (encoder_2 latent):
#   encode → Jobs A & B & diagnostic (all three in parallel, depend afterok on encode)
# Mirrors submit_all.sh; writes to outputs_instrument/ (physics outputs/ untouched).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ENC=$(sbatch --parsable "$HERE/job_encode_instrument.sh")
echo "encode  job: $ENC"

A=$(sbatch --parsable --dependency=afterok:"$ENC" "$HERE/job_uncond_instrument.sh")
echo "Job A   job: $A  (afterok:$ENC)"

B=$(sbatch --parsable --dependency=afterok:"$ENC" "$HERE/job_cond_instrument.sh")
echo "Job B   job: $B  (afterok:$ENC)"

D=$(sbatch --parsable --dependency=afterok:"$A":"$B" "$HERE/job_diag_instrument.sh")
echo "Diag    job: $D  (afterok:$A:$B)"

echo
echo "Watch:  squeue -u \$USER"
echo "Logs:   /work1/jeroenaudenaert/pablomer/logs/job.<jobid>.{out,err}"
echo "Outputs: $HERE/outputs_instrument/"
