"""Unconditional Flow Matching configuration for galaxy images.

This config is for training an unconditional flow matching model
that generates galaxy images without any conditioning.
"""

# Image configuration (after preprocessing)
IMAGE_SIZE = 96  # After center crop from 160x160
OUTPUT_DIM = IMAGE_SIZE * IMAGE_SIZE  # 96 * 96 = 9216 (flattened image)

# Model architecture parameters
VELOCITY_FIELD_TYPE = "unet"  # "mlp" or "unet"
USE_FILM = True  # Use FiLM (Feature-wise Linear Modulation) layers

# U-Net specific parameters (only used if VELOCITY_FIELD_TYPE == "unet")
UNET_CHANNELS = [32, 64, 128]  # Channel dimensions for U-Net encoder/decoder
NUM_RESIDUAL_LAYERS = 2  # Number of residual layers per U-Net block
T_EMBED_DIM = 40  # Time embedding dimension
Z_EMBED_DIM = 40  # Conditioning embedding dimension (unused for unconditional, but needed for network structure)

# MLP specific parameters (only used if VELOCITY_FIELD_TYPE == "mlp")
MLP_HIDDEN_DIMS = [128, 128]  # Hidden layer dimensions for MLP velocity field

# Flow matching parameters
N_INTEGRATION_STEPS = 100  # Number of ODE integration steps for sampling

# Training hyperparameters
BATCH_SIZE = 32  # Batch size for training
NUM_EPOCHS = 250  # Number of training epochs
LEARNING_RATE = 2e-4  # Learning rate

# Model save settings
SAVE_INTERVAL = 50  # Save model every N epochs
VISUALIZATION_INTERVAL = 5  # Show visualizations every N epochs

# Data configuration
SURVEY = 'HSC'  # Survey to use for data loading

# Pretrained model options
LOAD_PRETRAIN = False
# PATH_PRETRAIN = None  # Path to pretrained model checkpoint
