"""Galaxy Flow configuration

Imports shared defaults from double-encoder-model/config.py
and exposes a place to galaxy flow specific defaults without forking.
"""

import os
import sys

# Ensure shared config is importable
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))




SURVEY='HSC'

# Training hyperparameters
BATCH_SIZE = 128  # Reduced for quick test, use 128
NUM_EPOCHS = 250   # Just 1 epoch for quick test, use 50
LEARNING_RATE = 2e-4

# Model options
USE_FILM = True

# Concatenation options
USE_CONCATENATION = False
NUM_SAMPLES_CONCATENATION = 3  # Should match MULTI_NUM_FILTER_AUGS and MULTI_NUM_NUMBER_AUGS

# Attention pooling options
USE_ATTENTION = True
ATTENTION_NUM_LAYERS = 2
ATTENTION_NUM_HEADS = 4
ATTENTION_HIDDEN_DIM = 128


# Multi-sample batching
USE_MULTI_SAMPLES = True
MULTI_NUM_FILTER_AUGS = 5
MULTI_NUM_NUMBER_AUGS = 5





# Variable-length sequence options (between batches - use only for attention)
USE_VARIABLE_LENGTH = False  # Enable variable-length sequences between batches
MAX_FILTER_AUGS = 5  # Maximum number of filter augmentations
MAX_NUMBER_AUGS = 5  # Maximum number of number augmentations

# Validation: Variable-length sequences only make sense with attention
if USE_VARIABLE_LENGTH and not USE_ATTENTION:
    raise ValueError("USE_VARIABLE_LENGTH=True requires USE_ATTENTION=True. Variable-length sequences only work with attention-based pooling.")

# Pretrained model options
load_pretrain = False
# path_pretrain = "../flow_decoder/reconstruction_plots_v5_mnist/double_encoder_flow_model_mnist_200.pth"
# path_pretrain = '/root/work/tess-generative/flow_models/mnist/double-encoder-flow-mnist-v5-20251006_204716/double_encoder_flow_model_mnist_epoch_250_20251007_032221.pth'
# path_pretrain = '/root/work/tess-generative/flow_models/mnist/double-encoder-flow-mnist-v5-20251013_005213/double_encoder_flow_model_mnist_epoch_250_20251013_131027.pth'
# Model save settings
SAVE_INTERVAL = 50 # Save model every N epochsc - 50
VISUALIZATION_INTERVAL = 5  # Show visualizations every N epochs - 5

# Data paths
DATA_DIR = '/mnt/scratch/legacysurvey_hsc_crossmatched/data'
MODELS_DIR = '../models-galaxy/'
PLOTS_DIR = '../plots-galaxy/'
