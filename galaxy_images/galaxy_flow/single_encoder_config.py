"""Single-Encoder Flow Matching configuration for galaxy images.

This config is for training a conditional flow matching model
that generates galaxy images conditioned on encoded representations
from a single encoder (e.g., encode HSC images and generate Legacy images, or vice versa).
"""

# Image configuration (after preprocessing)
# IMAGE_SIZE = 96  # After center crop from 160x160
IMAGE_SIZE = 48
NUM_CHANNELS = 4  # Number of channels (g, r, i, z)
OUTPUT_DIM = NUM_CHANNELS * IMAGE_SIZE * IMAGE_SIZE  # 4 * 96 * 96 = 36864 (flattened multi-channel image)



ENCODER_TYPE = 'minuet'
# Encoder latent dim - used for both 'cnn' and 'resnet' encoder types
# For resnet: if set to 512, uses Identity layer (native ResNet output)
#            if set to other value, adds Linear layer to project from 512 to desired dim
# For minuet: this should match MINUET_BOTTLENECK_LENGTH
# ENCODER_LATENT_DIM = 512 # originally 40
# ENCODER_LATENT_DIM = 512
ENCODER_LATENT_DIM = 64

# Minuet encoder parameters (only used if ENCODER_TYPE == 'minuet')
MINUET_BOTTLENECK_LENGTH = ENCODER_LATENT_DIM  # Must match ENCODER_LATENT_DIM for decoder compatibility
MINUET_BOTTLENECK_DIM = 1  # Gets squeezed out in forward pass
MINUET_PATCH_SIZE = 3
MINUET_MODEL_DIM = 256
MINUET_NUM_HEADS = 4
MINUET_FF_DIM = 256
MINUET_NUM_LAYERS = 4
MINUET_DROPOUT = 0.1
MINUET_SELFATTN = False
MINUET_SINCOSIN = True

# Model architecture parameters
DECODER_TYPE = "latent"  # "latent" (uses encoder + latent z) or "concat" (direct image concatenation)
VELOCITY_FIELD_TYPE = "unet"  # "mlp" or "unet"
USE_FILM = True  # Use FiLM (Feature-wise Linear Modulation) layers

# U-Net specific parameters (only used if VELOCITY_FIELD_TYPE == "unet")
UNET_CHANNELS = [64, 128, 256]  # Channel dimensions for U-Net encoder/decoder (increased for better capacity)
# [64, 128, 256]
NUM_RESIDUAL_LAYERS = 2  # Number of residual layers per U-Net block
T_EMBED_DIM = 40  # Time embedding dimension
Z_EMBED_DIM = ENCODER_LATENT_DIM

# Concat decoder specific parameters (only used if DECODER_TYPE == "concat")
# Note: channels are in encoder order [small, ..., large] (e.g., [64, 128, 256])
CONCAT_UNET_CHANNELS = [64, 128, 256]  # Channel dimensions for concat decoder (encoder order: input to bottleneck)
CONCAT_COND_CHANNELS = NUM_CHANNELS  # Number of channels in conditioning images (usually same as NUM_CHANNELS)

# MLP specific parameters (only used if VELOCITY_FIELD_TYPE == "mlp")
MLP_HIDDEN_DIMS = [128, 128]  # Hidden layer dimensions for MLP velocity field

# Flow matching parameters
N_INTEGRATION_STEPS = 100  # Number of ODE integration steps for sampling
# Originally 250

# Training hyperparameters
BATCH_SIZE = 32  # Batch size for training
NUM_EPOCHS = 10_000  # Number of training epochs
LEARNING_RATE = 5e-4 / 8  # Learning rate
MAX_GRAD_NORM=1

# NUM_SAMPLES_PER_EPOCH=10280 # 2570
# NUM_SAMPLES_PER_EPOCH = 2570
NUM_SAMPLES_PER_EPOCH = 5140
WEIGHT_DECAY = 1e-5

# Model save settings
SAVE_INTERVAL = 200  # Save model every N epochs
VISUALIZATION_INTERVAL = 50  # Show visualizations every N epochs
PROFILE_FIRST_EPOCH = True  # Profile first epoch to show timing breakdown and FLOPs

# Data configuration
SURVEY = 'HSC'  # Survey to use for data loading
DATA_DIR = '/mnt/scratch/legacysurvey_hsc_crossmatched/data'

# Preprocessed data configuration (for faster loading)
USE_PREPROCESSED_DATA = True # Set to True to use preprocessed HDF5 format
# PREPROCESSED_HDF5_PATH = '/mnt/scratch/legacysurvey_hsc_crossmatched/preprocessed_hsc.h5'
# PREPROCESSED_HDF5_PATH = '/Users/pablom.perez/Desktop/data/legacysurvey_hsc_crossmatched/preprocessed_hsc.h5'
# PREPROCESSED_HDF5_PATH = '/Users/pablomercaderperez/Desktop/data/preprocessed/preprocessed_hsc_legacy.h5'
# PREPROCESSED_HDF5_PATH = '/Users/pablomercaderperez/Desktop/data/preprocessed/preprocessed_hsc_legacy_28x28.h5'
# PREPROCESSED_HDF5_PATH = '/Users/pablomercaderperez/Desktop/data/preprocessed/preprocessed_hsc_legacy_48x48.h5'
# PREPROCESSED_HDF5_PATH = '/Users/pablom.perez/Desktop/data/legacysurvey_hsc_crossmatched/preprocessed_hsc_legacy.h5'
# PREPROCESSED_HDF5_PATH = '/Users/pablom.perez/Desktop/data/legacysurvey_hsc_crossmatched/preprocessed_hsc_legacy_48x48.h5'

#csail clulster
# PREPROCESSED_HDF5_PATH = '/data/vision/billf/scratch/pablomer/legacysurvey_hsc/data/preprocessed_hsc_legacy_laptop.h5'
PREPROCESSED_HDF5_PATH = '/data/vision/billf/scratch/pablomer/legacysurvey_hsc/data/preprocessed_hsc_legacy_48x48_laptop.h5'

# Data loader options
LOAD_TO_MEMORY = True  # Whether to load all preprocessed images into memory
MAX_SAMPLES = None  # Maximum number of image pairs to use. Set to None to use all available pairs.
# MAX_SAMPLES=50

# Output directories
MODELS_DIR = './models-galaxy/'
PLOTS_DIR = './plots-galaxy/'

# Pretrained model options
LOAD_PRETRAIN = False
# PATH_PRETRAIN = None  # Path to pretrained model checkpoint

# Multi-GPU training (DDP)
# Number of GPUs to use for DDP training
# Set this to match the number of GPUs in your partition
# For example, if you have a partition with 4 GPUs but system sees 8, set NUM_GPUS = 4
NUM_GPUS = 8  # Number of GPUs for DDP (should match --nproc_per_node)

# To run with DDP:
# torchrun --nproc_per_node=8 -m galaxy_images.galaxy_flow.train_single_encoder_model
