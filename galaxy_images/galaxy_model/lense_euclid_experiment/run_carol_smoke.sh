#!/bin/bash
# Smoke test: 2 Carol lenses, 50 ODE steps. Run inside an interactive GPU job.
set -e

export TORCH_BLAS_PREFER_HIPBLASLT=0
export PYTHONPATH=/work1/jeroenaudenaert/pablomer/tess-generative:$PYTHONPATH

cd /work1/jeroenaudenaert/pablomer/tess-generative/galaxy_images/galaxy_model/lense_euclid_experiment

python eval_carol_legacy_to_euclid.py \
    --num-lenses 2 \
    --steps 50 \
    --out-dir outputs/carol_legacy_to_euclid_smoke

echo "---"
echo "Outputs:"
ls -la "$(realpath outputs/carol_legacy_to_euclid_smoke)"/
