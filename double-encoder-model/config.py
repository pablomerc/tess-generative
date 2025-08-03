"""
Configuration file for the Double Encoder Model
"""

import torch

# Device configuration
device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

# Dataset configuration
# DATASET_TYPE = 'mnist'  # Options: 'mnist' or 'fashion_mnist'
DATASET_TYPE = 'fashion_mnist'

# Model hyperparameters
LATENT_DIM = 16
NUMBER_ENCODER_LATENT_DIM = LATENT_DIM
FILTER_ENCODER_LATENT_DIM = LATENT_DIM

# Training hyperparameters
BATCH_SIZE = 256  # Reduced for quick test, use 128
NUM_EPOCHS = 50   # Just 1 epoch for quick test, use 50
LEARNING_RATE = 5e-4
BETA_KL = 0.01  # KL divergence weight
RECONSTRUCTION_WEIGHT = 1.0  # Reconstruction loss weight

# Data paths
DATA_DIR = '../data'
AUGMENTED_DATA_DIR = '../data/augmented'
MODELS_DIR = '../models'

# Augmentation parameters (from augment_mnist.py)
ROTATION_DEGREES = 90
ROTATION_STEP = 15  # Step size for rotation angles (degrees)
SCALE_RANGE = (0.5, 1.0)  # Zoom range - scale from 0.5x to 1.0x
TRANSLATE_RANGE = (0, 0)  # No translation for now

# Triplet creation parameters
MIN_ROTATION_DIFF = 0  # Minimum rotation difference between same digit samples
MAX_ROTATION_DIFF = 180  # Maximum rotation difference between same digit samples
MIN_SCALE_DIFF = 0.1  # Minimum scale difference between same digit samples

# Model save settings
SAVE_INTERVAL = 10  # Save model every N epochs
VISUALIZATION_INTERVAL = 10  # Show visualizations every N epochs

# Training settings
GRADIENT_CLIPPING = False  # Whether to use gradient clipping during training
