#!/bin/bash
# Submit 4 jobs: two parallel encodes, then two searches that fire when their encode finishes.
#
# Usage (from galaxy_model/):
#   bash neighbor_search_lenses/launch_lens_hsc_aion.sh

cd "$(dirname "$0")/.."  # galaxy_model/

JOB_ENC_A=$(sbatch --parsable neighbor_search_lenses/run_encode_hsc_extended.sh)
echo "Submitted encode-hsc-extended:  job $JOB_ENC_A"

JOB_ENC_B=$(sbatch --parsable neighbor_search_lenses/run_encode_overlap.sh)
echo "Submitted encode-overlap:        job $JOB_ENC_B"

JOB_SEARCH_A=$(sbatch --parsable --dependency=afterok:$JOB_ENC_A \
    neighbor_search_lenses/run_search_hsc_only.sh)
echo "Submitted search-hsc-only:       job $JOB_SEARCH_A (depends on $JOB_ENC_A)"

JOB_SEARCH_B=$(sbatch --parsable --dependency=afterok:$JOB_ENC_B \
    neighbor_search_lenses/run_search_combined.sh)
echo "Submitted search-combined:        job $JOB_SEARCH_B (depends on $JOB_ENC_B)"

echo ""
echo "Pipeline:"
echo "  [$JOB_ENC_A] encode HSC extended (~366k)  →  [$JOB_SEARCH_A] search HSC-only"
echo "  [$JOB_ENC_B] encode overlap (~206k total)  →  [$JOB_SEARCH_B] search combined"
