"""One-Hot Flow Matching configuration for galaxy images.

This config is for training a one-hot flow matching model
that generates galaxy images without any conditioning.

Adapted from unconditional_config.py
"""

# Image configuration (after preprocessing)
IMAGE_SIZE = 96  # After center crop from 160x160
NUM_CHANNELS = 4  # Number of channels (g, r, i, z)
OUTPUT_DIM = NUM_CHANNELS * IMAGE_SIZE * IMAGE_SIZE  # 4 * 96 * 96 = 36864 (flattened multi-channel image)

# Model architecture parameters
VELOCITY_FIELD_TYPE = "unet"  # "mlp" or "unet"
USE_FILM = True  # Use FiLM (Feature-wise Linear Modulation) layers

# U-Net specific parameters (only used if VELOCITY_FIELD_TYPE == "unet")
UNET_CHANNELS = [64, 128, 256]  # Channel dimensions for U-Net encoder/decoder (increased for better capacity)
# [64, 128, 256]
NUM_RESIDUAL_LAYERS = 2  # Number of residual layers per U-Net block
T_EMBED_DIM = 40  # Time embedding dimension
Z_EMBED_DIM = 40  # Conditioning embedding dimension (unused for unconditional, but needed for network structure)

# MLP specific parameters (only used if VELOCITY_FIELD_TYPE == "mlp")
MLP_HIDDEN_DIMS = [128, 128]  # Hidden layer dimensions for MLP velocity field

# Flow matching parameters
N_INTEGRATION_STEPS = 100  # Number of ODE integration steps for sampling (increased for better quality)
#just testing to see what happens if I reduce it, change back to 250?

# Training hyperparameters
BATCH_SIZE = 32  # Batch size for training
NUM_EPOCHS = 100  # Number of training epochs
LEARNING_RATE = 5e-4  # Learning rate
MAX_GRAD_NORM=1

NUM_SAMPLES_PER_EPOCH=10280
# NUM_SAMPLES_PER_EPOCH=1024
WEIGHT_DECAY = 1e-5

# Model save settings
SAVE_INTERVAL = 25  # Save model every N epochs
VISUALIZATION_INTERVAL = 4  # Show visualizations every N epochs
PROFILE_FIRST_EPOCH = True  # Profile first epoch to show timing breakdown and FLOPs

# # Data configuration
# SURVEY = 'HSC'  # Survey to use for data loading
# DATA_DIR = '/mnt/scratch/legacysurvey_hsc_crossmatched/data'

# Preprocessed data configuration (for faster loading)
USE_PREPROCESSED_DATA = True  # Set to True to use preprocessed HDF5 format
# HDF5 file with both HSC and Legacy images (required by HSC_Legacy_DataLoader_OneHot)
PREPROCESSED_HDF5_PATH = '/Users/pablom.perez/Desktop/data/legacysurvey_hsc_crossmatched/preprocessed_hsc_legacy.h5'

# Data loader options
LOAD_TO_MEMORY = True  # Whether to load all preprocessed images into memory

# Output directories
MODELS_DIR = './models-galaxy/'
PLOTS_DIR = './plots-galaxy/'

# Pretrained model options
LOAD_PRETRAIN = False
# PATH_PRETRAIN = None  # Path to pretrained model checkpoint
