# Flow V5 Entrypoint

This package provides a clean command-line entrypoint that wraps the existing implementation in `flow_decoder/VAE_flow_v5.py` without changing your current code. It lets you launch training with a clear, minimal interface.

## Usage

From the repository root:

```bash
python -m flow_v5.main --dataset mnist --epochs 50 --lr 2e-4 --batch_size 128
```

### Arguments
- `--dataset`: `mnist` or `fashion_mnist` (default: `mnist`)
- `--epochs`: number of epochs (default: 50)
- `--lr`: learning rate (default: 2e-4)
- `--batch_size`: batch size (default: 128)
- `--no_pretrain`: do not load pretrained weights
- `--pretrain_path`: path to checkpoint to load (default matches existing script)
- `--plots_dir`: directory for plots/checkpoints (default: top-level `flow_models/`)

## Downstream Task Evaluation

After training a model, you can evaluate the quality of learned representations by training downstream classifiers/regressors on the latent space:

### Classification Tasks

```bash
python -m flow_v5.downstream_tasks \
    --checkpoint <path_to_checkpoint> \
    --num_samples 10000 \
    --task classification \
    --epochs 50 \
    --lr 0.001
```

This will:
- Encode examples using the pretrained model
- Train classifiers to predict digit labels from `z_number` and rotation classes from `z_filter`
- Evaluate classification accuracy and create visualizations

### Regression Tasks

```bash
python -m flow_v5.downstream_tasks \
    --checkpoint <path_to_checkpoint> \
    --num_samples 10000 \
    --task regression \
    --epochs 50 \
    --lr 0.001
```

This will:
- Encode examples using the pretrained model
- Train regressors to predict rotation angles from latents
- Evaluate R2 scores, MSE, and MAE

### Multi-Sample Encoding

If your model was trained with multi-sample encoding:

```bash
python -m flow_v5.downstream_tasks \
    --checkpoint <path_to_checkpoint> \
    --multi_samples \
    --num_filter_augs 5 \
    --num_number_augs 5 \
    --task classification
```

### Arguments

- `--checkpoint`: Path to pretrained model checkpoint (required)
- `--num_samples`: Number of samples to encode (default: 10000)
- `--task`: `classification` or `regression` (default: `classification`)
- `--dataset`: `mnist` or `fashion_mnist` (defaults to config)
- `--multi_samples`: Use multi-sample encoding
- `--num_filter_augs`: Number of filter augmentations (for multi-sample)
- `--num_number_augs`: Number of number augmentations (for multi-sample)
- `--output_dir`: Output directory for results (default: auto-generated)
- `--epochs`: Number of training epochs (default: 50)
- `--lr`: Learning rate (default: 0.001)
- `--batch_size`: Batch size for downstream training (default: 64)

Results are saved to `flow_models/downstream_<dataset>_<task>_<timestamp>/` including:
- Training curves
- Summary visualizations
- Detailed results (text and CSV)
- Classification reports (for classification tasks)

## Notes
- This is a thin wrapper around your current training flow. All original files remain intact.
- You can incrementally move logic out of `flow_decoder/VAE_flow_v5.py` into smaller modules later (data, model, train, viz) without breaking the CLI.
