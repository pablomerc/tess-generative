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
- `--plots_dir`: directory for plots/checkpoints (default: `reconstruction_plots_v5_<dataset>`)

## Notes
- This is a thin wrapper around your current training flow. All original files remain intact.
- You can incrementally move logic out of `flow_decoder/VAE_flow_v5.py` into smaller modules later (data, model, train, viz) without breaking the CLI.
