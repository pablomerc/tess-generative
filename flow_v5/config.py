"""Flow V5 configuration

Imports shared defaults from double-encoder-model/config.py
and exposes a place to override V5-specific defaults without forking.
"""

import os
import sys

# Ensure shared config is importable
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DE_PATH = os.path.join(REPO_ROOT, 'double-encoder-model')
if DE_PATH not in sys.path:
    sys.path.insert(0, DE_PATH)

from config import *  # noqa: F401,F403

# Optional V5-specific overrides (keep minimal to avoid drift)
# Example: Different default dataset for v5
# DATASET_TYPE = os.environ.get('FLOW_V5_DATASET', DATASET_TYPE)

# You can also override integration steps or architecture defaults here if needed
# For now, we keep architecture choices in code to avoid duplicating too much config.
