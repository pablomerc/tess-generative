# Galaxy Images Models

This directory contains models for cross-instrument prediction with galaxy images, similar to the flow_v5 architecture but adapted for multi-channel astronomical images.

## Structure

```
galaxy_images/
├── galaxy_triplets.py          # Data loading (already exists)
├── galaxy_vae/                 # VAE models for galaxy images
│   ├── __init__.py
│   ├── config.py              # Configuration for VAE
│   ├── encoders.py            # Galaxy-specific encoders
│   ├── decoder.py             # Galaxy-specific decoder
│   ├── model.py               # Complete VAE model
│   ├── train.py               # Training script
│   └── viz.py                 # Visualization utilities
├── galaxy_flow/               # Flow matching models (future)
│   └── (to be created after VAE works)
└── tests/                     # Test files
```

## Approach

1. **Start with VAE**: Build a simple VAE first to ensure the data pipeline and basic reconstruction works
2. **Then Flow Matching**: Once VAE works, adapt the flow matching architecture from flow_v5

## Key Differences from MNIST Models

- **Image size**: 160x160 (vs 28x28)
- **Channels**: 12-13 channels (flux, ivar, mask, optionally object_mask) vs 1 channel
- **Semantics**: Cross-instrument prediction (legacysurvey ↔ hsc) vs cross-augmentation
- **Data format**: Multi-band astronomical images with uncertainty information

## Usage

### Training VAE

```bash
python -m galaxy_images.galaxy_vae.train --epochs 50 --batch_size 32
```

### Training Flow Matching (after VAE works)

```bash
python -m galaxy_images.galaxy_flow.train --epochs 100 --batch_size 32
```
