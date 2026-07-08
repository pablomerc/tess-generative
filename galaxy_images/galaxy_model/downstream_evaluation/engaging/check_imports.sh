#!/bin/bash
# Verify all module-side imports work in torchenv before resubmitting sbatch.
# Run this on your interactive mit_normal node:
#   bash check_imports.sh
set -e
source ~/.bashrc
conda activate torchenv
cd /home/pablomer/orcd/pool/tess-generative

python - <<'PY'
import importlib, sys
print("Python:", sys.executable)
print()
print("--- 3rd-party deps the model files import ---")
for m in ['wandb','geomloss','umap','timm','diffusers','pyarrow','h5py','astropy','scienceplots']:
    try:
        mod = importlib.import_module(m)
        print(f"  OK  {m:14s} {getattr(mod, '__version__', '?')}")
    except Exception as e:
        print(f"  FAIL {m:14s} {e.__class__.__name__}: {e}")
print()
print("--- LightningModule classes (full forward-compat check) ---")
import importlib.util
from pathlib import Path
gm = Path('/home/pablomer/orcd/pool/tess-generative/galaxy_images/galaxy_model')
sys.path.insert(0, '/home/pablomer/orcd/pool/tess-generative')
targets = [
    ('double_train_fm_neighbors.py', 'ConditionalFlowMatchingModule'),
    ('hierarchical_attention/double_train_fm_neighbors_hier_global_ins.py', 'HierarchicalGlobalInstrumentFlowMatchingModule'),
    ('single_encoder_ablation/model.py', 'SingleEncoderFlowMatchingModule'),
]
for rel, cls in targets:
    path = gm / rel
    spec = importlib.util.spec_from_file_location(rel.replace('/', '_').replace('.py',''), path)
    try:
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        getattr(mod, cls)
        print(f"  OK  {rel}::{cls}")
    except Exception as e:
        print(f"  FAIL {rel}::{cls}  {e.__class__.__name__}: {e}")
PY
