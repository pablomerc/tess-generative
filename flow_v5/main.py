import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser(description="Run Double Encoder Flow Matching v5 training")

    # Resolve repository root and add paths for imports
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, repo_root)
    de_path = os.path.join(repo_root, 'double-encoder-model')
    if de_path not in sys.path:
        sys.path.insert(0, de_path)

    # Import v5 config for defaults (which imports shared defaults)
    from flow_v5 import config as cfg

    parser.add_argument("--dataset", choices=["mnist", "fashion_mnist"], default=cfg.DATASET_TYPE,
                        help="Dataset type")
    parser.add_argument("--epochs", type=int, default=cfg.NUM_EPOCHS, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=cfg.LEARNING_RATE, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=cfg.BATCH_SIZE, help="Batch size")
    parser.add_argument("--no_pretrain", action="store_true", help="Do not load pretrained weights")
    parser.add_argument("--pretrain_path", type=str, default=None,
                        help="Path to pretrained checkpoint")
    parser.add_argument("--plots_dir", type=str, default=None, help="Directory to save plots and checkpoints")

    args = parser.parse_args()

    # Import helpers
    from flow_v5.data import make_triplet_creator
    from flow_v5.model import build_model
    from flow_v5.train import train as train_v5
    from flow_v5.viz import debug_normalization
    import torch
    import wandb
    from datetime import datetime
    import re

    # Configure from args (which default to cfg)
    dataset_type = args.dataset
    num_epochs = args.epochs
    learning_rate = args.lr
    batch_size = args.batch_size
    load_pretrain = (not args.no_pretrain) if args.no_pretrain is not None else cfg.load_pretrain
    path_pretrain = args.pretrain_path or cfg.path_pretrain
    # Default to top-level folder for artifacts
    plots_dir = args.plots_dir or "flow_models"

    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

    # Triplet creator
    triplet_creator = make_triplet_creator(dataset_type=dataset_type)
    triplet_creator.get_dataset_info()

    # Debug normalization once
    debug_normalization(triplet_creator, device)

    os.makedirs(plots_dir, exist_ok=True)

    # Model dims from config
    number_latent_dim = cfg.NUMBER_ENCODER_LATENT_DIM
    filter_latent_dim = cfg.FILTER_ENCODER_LATENT_DIM

    model = build_model(device=device)

    # Optional pretrain
    start_epoch = 0
    if load_pretrain:
        if os.path.exists(path_pretrain):
            try:
                print(f"Loading pretrained weights from: {path_pretrain}")
                checkpoint = torch.load(path_pretrain, map_location=device)
                state_dict = checkpoint.get('model_state_dict', checkpoint)
                missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
                print(f"Loaded pretrained weights. Missing keys: {len(missing_keys)}, Unexpected keys: {len(unexpected_keys)}")
                # Determine how many epochs were previously trained
                if isinstance(checkpoint, dict):
                    if 'epochs_trained' in checkpoint:
                        start_epoch = int(checkpoint['epochs_trained'])
                    elif 'train_losses' in checkpoint and isinstance(checkpoint['train_losses'], list):
                        # fall back to the length of losses
                        start_epoch = len(checkpoint['train_losses'])
                if start_epoch == 0:
                    # Try parsing from filename pattern ..._epoch_{N}_...
                    m = re.search(r"_epoch_(\d+)_", os.path.basename(path_pretrain) or "")
                    if m:
                        try:
                            start_epoch = int(m.group(1))
                        except Exception:
                            pass
                print(f"Resuming training with start_epoch={start_epoch} (cumulative epoch counter)")
            except Exception as e:
                print(f"WARNING: Failed to load pretrained weights from {path_pretrain}: {e}")
                print("Proceeding with training from scratch.")
        else:
            print(f"WARNING: Pretrain requested but checkpoint not found: {path_pretrain}")
            print("Proceeding with training from scratch.")
    else:
        print("Training from scratch (pretrain disabled).")

    # wandb init
    wandb.init(
        project="tess-generative",
        name=f"double-encoder-flow-{dataset_type}-v5-{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        config={
            "dataset_type": dataset_type,
            "num_epochs": num_epochs,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "device": device,
            "start_epoch": start_epoch,
        }
    )
    wandb.watch(model, log="all")

    # Train
    train_v5(model, triplet_creator, num_epochs=num_epochs, lr=learning_rate, plots_dir=plots_dir, start_epoch=start_epoch)

    wandb.finish()


if __name__ == "__main__":
    main()
