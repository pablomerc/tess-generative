#!/bin/bash
# Full run: all 15 Carol lenses with both Legacy and Euclid, 100 ODE steps.
set -e

export TORCH_BLAS_PREFER_HIPBLASLT=0
export PYTHONPATH=/work1/jeroenaudenaert/pablomer/tess-generative:$PYTHONPATH

cd /work1/jeroenaudenaert/pablomer/tess-generative/galaxy_images/galaxy_model/lense_euclid_experiment

python eval_carol_legacy_to_euclid.py \
    --steps 100 \
    --out-dir outputs/carol_legacy_to_euclid

echo "---"
echo "Outputs:"
ls -la "$(realpath outputs/carol_legacy_to_euclid)"/
