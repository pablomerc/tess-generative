"""
Configuration for Galaxy VAE models
"""

import torch

# Device configuration
device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

# Galaxy image configuration
IMAGE_SIZE = 160  # Galaxy images are 160x160
NUM_BANDS = 4  # Number of bands (g, r, i, z typically)
USE_IVAR = False  # Whether to include ivar (adds 4 channels: 4-7)
USE_FLUX_MASK = False  # Whether to include flux mask (adds 4 channels: 8-11)
USE_OBJECT_MASK = True  # Whether to include object mask (adds 1 channel: 12)

# Channel structure from triplet loader (always provides all):
# - flux: NUM_BANDS channels (0-3) - always included
# - ivar: NUM_BANDS channels (4-7) - included if USE_IVAR=True
# - mask: NUM_BANDS channels (8-11) - included if USE_FLUX_MASK=True
# - object_mask: 1 channel (12) - included if USE_OBJECT_MASK=True
NUM_CHANNELS = NUM_BANDS  # Start with flux (4 channels)
if USE_IVAR:
    NUM_CHANNELS += NUM_BANDS  # Add ivar (4 channels)
if USE_FLUX_MASK:
    NUM_CHANNELS += NUM_BANDS  # Add mask (4 channels)
if USE_OBJECT_MASK:
    NUM_CHANNELS += 1  # Add object mask (1 channel)

# Model hyperparameters
LATENT_DIM = 32  # We used 16 for the MNIST dataset, these images are larger so we can start with 32
NUMBER_ENCODER_LATENT_DIM = LATENT_DIM  # Encodes galaxy identity (same galaxy, different instrument)
FILTER_ENCODER_LATENT_DIM = LATENT_DIM  # Encodes instrument characteristics (different galaxy, same instrument)

# Training hyperparameters
BATCH_SIZE = 32  # Smaller batch size for larger images
NUM_EPOCHS = 300
NUM_SAMPLES_PER_EPOCH = 1000
NUM_BATCHES_PER_EPOCH = NUM_SAMPLES_PER_EPOCH // BATCH_SIZE
LEARNING_RATE = 5e-4
BETA_KL = 0.01  # KL divergence weight
RECONSTRUCTION_WEIGHT = 1.0  # Reconstruction loss weight

# Data paths
DATA_DIR = '/mnt/scratch/legacysurvey_hsc_crossmatched/data'
MODELS_DIR = '../models-galaxy/'
PLOTS_DIR = '../plots-galaxy/'

# Model save settings
SAVE_INTERVAL = 10  # Save model every N epochs
VISUALIZATION_INTERVAL = 5  # Show visualizations every N epochs

# Training settings
GRADIENT_CLIPPING = True
MAX_GRAD_NORM = 1.0

# Early stopping configuration
EARLY_STOPPING_PATIENCE = 20
EARLY_STOPPING_MIN_DELTA = 0.01
