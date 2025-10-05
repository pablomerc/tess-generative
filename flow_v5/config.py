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

DATASET_TYPE='fashion_mnist'
# Training hyperparameters
BATCH_SIZE = 128  # Reduced for quick test, use 128
NUM_EPOCHS = 250   # Just 1 epoch for quick test, use 50
LEARNING_RATE = 2e-4

# Multi-sample batching
USE_MULTI_SAMPLES = False
MULTI_NUM_FILTER_AUGS = 5
MULTI_NUM_NUMBER_AUGS = 5


# Pretrained model options
load_pretrain = False
# path_pretrain = "../flow_decoder/reconstruction_plots_v5_mnist/double_encoder_flow_model_mnist_200.pth"

# Model save settings
SAVE_INTERVAL = 50  # Save model every N epochsc - 10
VISUALIZATION_INTERVAL = 5  # Show visualizations every N epochs - 10

# Model options
USE_FILM = True
