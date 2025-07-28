# Double Encoder Model for MNIST Disentanglement

This project implements a novel approach to disentangling content (digit identity) and style (augmentation/filter) representations using a dual encoder architecture trained on MNIST triplets.

## Overview

The goal is to learn separate latent representations for:
1. **Digit Identity**: What digit is being represented (0-9)
2. **Augmentation Style**: How the digit is transformed (rotation, etc.)

### Architecture

The model consists of three main components:

1. **Number Encoder**: Takes a digit with one augmentation and learns to encode the digit identity
2. **Filter Encoder**: Takes a different digit with the same augmentation and learns to encode the augmentation style
3. **Joint Decoder**: Combines both encodings to reconstruct the ground truth image

### Training Pipeline

For each training step, we create a triplet of images:
- **Ground Truth**: Original digit with specific augmentation (e.g., 7 with 30° rotation)
- **Different Digit**: Different digit with same augmentation (e.g., 3 with 30° rotation) → Input to Filter Encoder
- **Same Digit**: Same digit with different augmentation (e.g., 7 with -45° rotation) → Input to Number Encoder

The model learns to reconstruct the ground truth by combining the digit identity from the Number Encoder and the augmentation style from the Filter Encoder.

### Augmentation Pipeline

The augmentation is done with `augment_mnist.py` and it consists of:
1. **Upsample**: Resize to 56×56 using bilinear interpolation
2. **Rotate**: Apply specific rotation angle with bilinear interpolation
3. **Downsample**: Resize back to 28×28 using Lanczos interpolation
4. **Convert**: Transform to tensor format

This approach preserves image quality and reduces artifacts during rotation.

## File Structure

```
double-encoder-model/
├── config.py                    # Configuration parameters
├── triplet_creation.py          # Creates training triplets
├── encoder_architectures.py     # Number and Filter encoders
├── decoder.py                   # Joint decoder
├── utils.py                     # Loss functions, metrics, visualization
├── training_pipeline.py         # Main training script
└── README.md                    # This file
```

## Usage

### 1. Test Individual Components

Test the triplet creation:
```bash
cd tess-generative/double-encoder-model
python triplet_creation.py
```

Test the encoder architectures:
```bash
python encoder_architectures.py
```

Test the decoder:
```bash
python decoder.py
```

Test the utilities:
```bash
python utils.py
```

### 2. Run Training

Start the complete training pipeline:
```bash
python training_pipeline.py
```

### 3. Configuration

Modify `config.py` to adjust:
- **Model hyperparameters**: Latent dimensions, learning rate
- **Training parameters**: Batch size, number of epochs, β (KL weight)
- **Augmentation settings**: Rotation angles, minimum/maximum differences
- **Save intervals**: Model saving and visualization frequency

Key configurable parameters:
- `BETA_KL`: Controls KL divergence weight (crucial for VAE training)
- `RECONSTRUCTION_WEIGHT`: Weight for reconstruction loss
- `MIN_ROTATION_DIFF`/`MAX_ROTATION_DIFF`: Controls rotation diversity in triplets

## Model Architecture Details

### Encoders
Both encoders use the same CNN architecture based on **ConditionalConvVAE**:
- 2 convolutional layers with 3×3 kernels and padding=1
  - `nn.Conv2d(1, 64, 3, 1, 1)` → `nn.Conv2d(64, 128, 3, 1, 1)`
- Maintains spatial dimensions (28×28) throughout
- Flatten to 128×28×28 = 100,352 features
- Project to latent dimension (default: 32)

### Decoder
The joint decoder:
- Concatenates number and filter encodings (64 total dimensions)
- Projects to 128×28×28 features
- 2 transposed convolutional layers with 3×3 kernels
  - `nn.ConvTranspose2d(128, 64, 3, 1, 1)` → `nn.ConvTranspose2d(64, 1, 3, 1, 1)`
- Sigmoid activation for output (ensures [0,1] range)

### Loss Function
Total loss combines:
- **Reconstruction Loss**: MSE between reconstructed and ground truth images
- **KL Divergence**: Regularization for both encoders' latent spaces

```
Total Loss = Reconstruction Weight × MSE + β × (KL_number + KL_filter)
```

The β parameter is crucial for balancing reconstruction quality vs. latent space structure.

## Expected Results

After training, the model should:

1. **Learn Disentangled Representations**:
   - Number encoder should cluster by digit identity regardless of augmentation
   - Filter encoder should cluster by augmentation style regardless of digit

2. **Enable Controlled Generation**:
   - Combine any digit identity with any augmentation style
   - Generate new digits with specific rotations

3. **Good Reconstruction Quality**:
   - Low MSE and high PSNR on test set
   - Visually similar reconstructions to ground truth

## Visualization

The training pipeline automatically generates:
- **Triplet Reconstructions**: Shows original vs reconstructed images
- **Latent Space Plots**: t-SNE visualization of encoder outputs
- **Training Curves**: Loss and metric progression over time
- **Generation Tests**: Combining different number and filter encodings

## Model Checkpoints

Models are saved with comprehensive metadata including:
- Model state and optimizer state
- Training epoch and loss
- All configuration parameters (β, learning rate, batch size, etc.)
- Architecture details (latent dimensions)

This ensures full reproducibility and easy experiment tracking.

## Dependencies

- PyTorch
- torchvision
- numpy
- matplotlib
- scikit-learn (for t-SNE visualization)
- seaborn

## Future Extensions

1. **Additional Augmentations**: Scale, translation, noise
2. **Conditional Generation**: Generate specific digits with specific styles
3. **Interpolation**: Smooth transitions between different styles
4. **Multi-class**: Extend to other datasets (Fashion-MNIST, CIFAR)
5. **Adversarial Training**: Add discriminator for better quality
6. **β-scheduling**: Gradually increase β during training for better convergence

## Research Context

This approach is inspired by:
- **β-VAE**: For disentangled representations
- **Style Transfer**: For separating content and style
- **Contrastive Learning**: For learning meaningful representations
- **Conditional VAEs**: For controlled generation

The key innovation is using the triplet structure to force the model to learn separate representations for content and style through the reconstruction objective.
