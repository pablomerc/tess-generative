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

DATASET_TYPE='mnist'
# Training hyperparameters
BATCH_SIZE = 128  # Reduced for quick test, use 128
NUM_EPOCHS = 20   # Just 1 epoch for quick test, use 50
LEARNING_RATE = 2e-4

# Multi-sample batching
USE_MULTI_SAMPLES = True
MULTI_NUM_FILTER_AUGS = 5
MULTI_NUM_NUMBER_AUGS = 5

# Concatenation options
USE_CONCATENATION = False
NUM_SAMPLES_CONCATENATION = 3  # Should match MULTI_NUM_FILTER_AUGS and MULTI_NUM_NUMBER_AUGS

# Attention pooling options
USE_ATTENTION = True
ATTENTION_NUM_LAYERS = 2
ATTENTION_NUM_HEADS = 4
ATTENTION_HIDDEN_DIM = 128


# Pretrained model options
load_pretrain = True
# path_pretrain = "../flow_decoder/reconstruction_plots_v5_mnist/double_encoder_flow_model_mnist_200.pth"
# path_pretrain = '/root/work/tess-generative/flow_models/mnist/double-encoder-flow-mnist-v5-20251006_204716/double_encoder_flow_model_mnist_epoch_250_20251007_032221.pth'
path_pretrain = '/root/work/tess-generative/flow_models/mnist/double-encoder-flow-mnist-v5-20251013_005213/double_encoder_flow_model_mnist_epoch_250_20251013_131027.pth'
# Model save settings
SAVE_INTERVAL = 10 # Save model every N epochsc - 50
VISUALIZATION_INTERVAL = 5  # Show visualizations every N epochs - 5

# Model options
USE_FILM = True
