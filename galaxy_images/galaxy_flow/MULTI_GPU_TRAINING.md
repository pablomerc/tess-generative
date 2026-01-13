# Multi-GPU Training Guide

This guide explains how to use multiple GPUs for training the single encoder flow matching model.

## Quick Answer

**Yes, it's easy to use PyTorch for multi-GPU training!** Two options:

1. **DataParallel (DP)** - Easiest, minimal code changes
2. **DistributedDataParallel (DDP)** - Better performance, more setup

## Expected Speedups

### DataParallel
- **2 GPUs**: ~1.5-1.8x speedup
- **4 GPUs**: ~2.5-3x speedup
- **Limitation**: GIL bottleneck limits scaling efficiency

### DistributedDataParallel (DDP)
- **2 GPUs**: ~1.8-1.95x speedup (near-linear)
- **4 GPUs**: ~3.5-3.9x speedup (near-linear)
- **Best for**: Production training with 2+ GPUs

## Usage

### Option 1: DataParallel (Recommended for Quick Start)

Simply set in `single_encoder_config.py`:

```python
USE_MULTI_GPU = 'auto'  # Automatically uses all available GPUs
```

Or force it:
```python
USE_MULTI_GPU = 'dp'  # Force DataParallel
```

Then run normally:
```bash
python train_single_encoder_model.py
```

The script will automatically:
- Detect multiple GPUs
- Wrap the model with `nn.DataParallel`
- Distribute batches across GPUs
- Handle model unwrapping for saving/visualization

### Option 2: DistributedDataParallel (Best Performance)

**Step 1**: Modify the main block to use DDP (or add a command-line flag)

**Step 2**: Launch with `torchrun`:

```bash
# For 2 GPUs
torchrun --nproc_per_node=2 train_single_encoder_model.py

# For 4 GPUs
torchrun --nproc_per_node=4 train_single_encoder_model.py

# For 8 GPUs
torchrun --nproc_per_node=8 train_single_encoder_model.py
```

**Note**: DDP requires the script to be modified to call `train_ddp()` instead of `train()`. The function is already implemented in the training script.

## Key Implementation Details

### DataParallel Changes Made:
1. Model wrapping: `model = nn.DataParallel(model)`
2. Device handling: Uses `cuda:0` as primary device
3. Model unwrapping: `model.module` for saving/visualization
4. Automatic batch distribution across GPUs

### DDP Changes Made:
1. Process group initialization
2. Per-process device assignment (`cuda:rank`)
3. Loss synchronization across processes
4. Learning rate scaling (multiply by world_size)
5. Only rank 0 does logging/checkpointing

## Performance Considerations

1. **Batch Size**: With multi-GPU, effective batch size = `BATCH_SIZE × num_gpus`
   - You may want to reduce per-GPU batch size to maintain same effective batch size
   - Or increase effective batch size for better training stability

2. **Memory**: Each GPU needs enough memory for the model + batch
   - Current config: `BATCH_SIZE = 32` per GPU
   - With 4 GPUs, effective batch size = 128

3. **Data Loading**: The current data loader should work fine with both DP and DDP
   - DP: Single process, batches split across GPUs automatically
   - DDP: Each process gets different batches

## Troubleshooting

### DataParallel Issues:
- **Out of Memory**: Reduce `BATCH_SIZE` in config
- **Slow**: Normal for DP due to GIL bottleneck - consider DDP

### DDP Issues:
- **Hanging**: Check that all processes can communicate (NCCL backend)
- **Different results**: Ensure random seeds are set per process
- **File conflicts**: Only rank 0 should write checkpoints (already handled)

## Current Model Suitability

Your model architecture is **perfectly suited** for multi-GPU training:
- ✅ Standard `nn.Module` structure
- ✅ No shared state between forward passes
- ✅ Batch-independent operations
- ✅ Standard loss computation

The encoder-decoder structure with flow matching decoder works seamlessly with both DP and DDP.
