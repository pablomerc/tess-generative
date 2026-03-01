"""
Prepare neighbors downstream dataset: use NeighborsSimpleDataset (single HDF5 with
images + metadata), extract embeddings (trained + untrained) with same model types as
prepare_mmu, save to HDF5. Labels come from the dataset metadata (all non-excluded columns).

Run from galaxy_model/ or downstream_evaluation/:
  python downstream_evaluation/prepare_neighbors.py test   # dataloader only
  python downstream_evaluation/prepare_neighbors.py prepare  # full pipeline
"""
import sys
from pathlib import Path

_here = Path(__file__).resolve().parent
_src = _here.parent
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from neighbors import NeighborsSimpleDataset

# Neighbors HDF5 (single file with images_hsc, images_legacy, source_type, and metadata columns)
NEIGHBORS_HDF5 = "/data/vision/billf/scratch/pablomer/data/neighbours_v2.h5"

# NUM_EXAMPLES = 8192  # cap for downstream (None = use full dataset)
NUM_EXAMPLES = 4096
SEED = 42  # for reproducible subset order and/or DataLoader shuffle
SHUFFLE_LOADER = True  # shuffle batches (with SEED); if False, use seeded perm of indices for order
IMAGE_SIZE = 48

# Same model configs as prepare_mmu
HSC_MODEL_CONFIGS = [
    ("zdim16_geom_neighbors", "/data/vision/billf/scratch/pablomer/projects/tess-generative/galaxy-flow-matching-neighbours/03mwpijx/checkpoints/latest-step=step=75000.ckpt", "double_train_fm_neighbors.py"),
    ("zdim16_nogeom_neighbors", "/data/vision/billf/scratch/pablomer/projects/tess-generative/galaxy-flow-matching-neighbours/g2g9kvr4/checkpoints/latest-step=step=75000.ckpt", "double_train_fm_neighbors.py"),
    ("zdim16_geom_old_dataloader", "/data/vision/billf/scratch/pablomer/projects/tess-generative/galaxy-flow-matching/wu1csh99/checkpoints/latest-step=step=75000.ckpt", "double_train_fm.py"),
]

OUTPUT_DOWNSTREAM_UNTRAINED_H5 = _here / "downstream_neighbors_untrained.h5"

EMBEDDING_NAMES = (
    "hsc_encoder1",
    "hsc_encoder2",
    "legacy_encoder1",
    "legacy_encoder2",
    "hsc_legacy_encoder1",
    "hsc_legacy_encoder2",
)


def collate_neighbors_simple(batch):
    """Collate (img_hsc, img_legacy, metadata) into batched tensors + list of dicts."""
    hsc = torch.stack([b[0] for b in batch])
    leg = torch.stack([b[1] for b in batch])
    meta = [b[2] for b in batch]
    return hsc, leg, meta


def _labels_from_metadata_list(metadata_list):
    """
    Build labels dict from list of metadata dicts (from NeighborsSimpleDataset).
    Includes keys whose values are numeric (int/float) and stack into arrays.
    Returns (labels_n, label_columns).
    """
    if not metadata_list:
        return {}, []

    keys = list(metadata_list[0].keys())
    labels_n = {}
    valid_columns = []

    for col in keys:
        try:
            vals = [m[col] for m in metadata_list]
            arr = np.array(vals)
            if arr.dtype.kind in "iuflb" and arr.shape[0] == len(metadata_list):
                labels_n[col] = arr.astype(np.float32) if arr.dtype.kind in "f" else arr
                valid_columns.append(col)
            elif arr.dtype.kind in "iuflb" and arr.ndim == 2 and arr.shape[0] == len(metadata_list):
                labels_n[col] = arr.astype(np.float32)
                valid_columns.append(col)
            else:
                pass
        except (TypeError, ValueError):
            pass

    return labels_n, valid_columns


def _load_model(checkpoint_path, module_filename):
    """Load the double-encoder model from checkpoint."""
    import importlib.util

    module_name = module_filename.replace(".py", "").replace(".", "_")
    spec = importlib.util.spec_from_file_location(
        module_name,
        _src / module_filename,
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    model = mod.ConditionalFlowMatchingModule.load_from_checkpoint(
        checkpoint_path, map_location="cpu"
    )
    model.eval()
    torch.set_grad_enabled(False)
    return model


def _load_untrained_model():
    """Create an untrained model with same architecture as double_train_fm_neighbors.py."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "double_train_fm_neighbors",
        _src / "double_train_fm_neighbors.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    model = mod.ConditionalFlowMatchingModule(
        in_channels=4,
        cond_channels=4,
        image_size=IMAGE_SIZE,
        model_channels=128,
        channel_mult=(1, 2, 4, 4),
        cross_attention_dim=16,
        pretrained_encoder=False,
        concat_conditioning=False,
        lr=1e-4,
        num_sample_images=10,
        num_mse_images=32,
        num_integration_steps=250,
        lambda_generative=1.0,
        lambda_geometric=0.3,
        mask_center=False,
    )
    model.eval()
    torch.set_grad_enabled(False)
    return model


def generate_embeddings(model, dataset, device, batch_size=256, shuffle=False, seed=None):
    """
    Run encoder_1 and encoder_2 on HSC and Legacy; return 6 embedding arrays
    and the collected metadata list (for building labels).
    If shuffle=True, use a seeded generator so order is reproducible.
    """
    loader_kw = dict(
        batch_size=batch_size,
        num_workers=0,
        collate_fn=collate_neighbors_simple,
    )
    if shuffle:
        g = torch.Generator()
        if seed is not None:
            g.manual_seed(seed)
        loader_kw["shuffle"] = True
        loader_kw["generator"] = g
    else:
        loader_kw["shuffle"] = False

    loader = DataLoader(dataset, **loader_kw)
    hsc_1, hsc_2, leg_1, leg_2 = [], [], [], []
    metadata_collected = []

    with torch.no_grad():
        for hsc_im, leg_im, meta_list in loader:
            metadata_collected.extend(meta_list)
            h, l = hsc_im.to(device), leg_im.to(device)
            hsc_1.append(model.encoder_1(h).cpu())
            hsc_2.append(model.encoder_2(h).cpu())
            leg_1.append(model.encoder_1(l).cpu())
            leg_2.append(model.encoder_2(l).cpu())

    e1 = torch.cat(hsc_1, dim=0).flatten(start_dim=1)
    e2 = torch.cat(hsc_2, dim=0).flatten(start_dim=1)
    e3 = torch.cat(leg_1, dim=0).flatten(start_dim=1)
    e4 = torch.cat(leg_2, dim=0).flatten(start_dim=1)
    e5 = torch.cat(
        [
            torch.cat([h.flatten(start_dim=1), l.flatten(start_dim=1)], dim=1)
            for h, l in zip(hsc_1, leg_1)
        ],
        dim=0,
    )
    e6 = torch.cat(
        [
            torch.cat([h.flatten(start_dim=1), l.flatten(start_dim=1)], dim=1)
            for h, l in zip(hsc_2, leg_2)
        ],
        dim=0,
    )
    return (
        e1.numpy(),
        e2.numpy(),
        e3.numpy(),
        e4.numpy(),
        e5.numpy(),
        e6.numpy(),
        metadata_collected,
    )


def save_downstream_h5(path, embeddings_dict, labels_n, label_columns):
    """Save 6 embedding datasets and labels to HDF5."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        for name in EMBEDDING_NAMES:
            arr = embeddings_dict[name]
            f.create_dataset(name, data=arr, compression="gzip", compression_opts=4)
        for col in label_columns:
            data = labels_n[col]
            f.create_dataset(
                f"labels/{col}", data=data, compression="gzip", compression_opts=4
            )
        f.attrs["embedding_names"] = list(EMBEDDING_NAMES)
        n = len(labels_n[label_columns[0]]) if label_columns else embeddings_dict[EMBEDDING_NAMES[0]].shape[0]
        f.attrs["num_examples"] = n
        f.attrs["label_columns"] = list(label_columns)
    print(f"Saved: {path} (6 embeddings + {len(label_columns)} labels)")


def prepare_downstream_dataset(batch_size=256):
    """
    Generate embeddings with pretrained (and untrained) models for neighbors subset,
    use metadata from NeighborsSimpleDataset as labels, save to downstream_neighbors_*.h5.
    Order is reproducible via SEED; use SHUFFLE_LOADER=True for shuffled batches (labels built from first run).
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(f"Loading NeighborsSimpleDataset from {NEIGHBORS_HDF5}...")
    full_dataset = NeighborsSimpleDataset(hdf5_path=NEIGHBORS_HDF5)
    n_total = len(full_dataset)
    n_use = min(NUM_EXAMPLES, n_total) if NUM_EXAMPLES is not None else n_total

    if SHUFFLE_LOADER:
        dataset = Subset(full_dataset, range(n_use))
        print(f"Using subset: {n_use} examples (DataLoader shuffle=True, seed={SEED})")
    else:
        gen = torch.Generator().manual_seed(SEED)
        perm = torch.randperm(n_total, generator=gen)
        indices = perm[:n_use].tolist()
        dataset = Subset(full_dataset, indices)
        print(f"Using subset: {n_use} examples (seeded perm order, seed={SEED})")

    # Labels: same order as embeddings. If shuffle=True we get order from first embedding run.
    if not SHUFFLE_LOADER:
        print("Collecting metadata for labels...")
        meta_list = [dataset[i][2] for i in range(len(dataset))]
        labels_n, label_columns = _labels_from_metadata_list(meta_list)
        print(f"Built {len(label_columns)} label columns from metadata")

    saved_paths = []
    for suffix, checkpoint_path, module_filename in HSC_MODEL_CONFIGS:
        output_h5 = _here / f"downstream_neighbors_{suffix}.h5"
        print(f"Loading model: {suffix} from {checkpoint_path}")
        model = _load_model(checkpoint_path, module_filename).to(device)
        print(f"Generating embeddings for {suffix}...")
        e1, e2, e3, e4, e5, e6, metadata_collected = generate_embeddings(
            model, dataset, device, batch_size,
            shuffle=SHUFFLE_LOADER, seed=SEED if SHUFFLE_LOADER else None,
        )
        if SHUFFLE_LOADER and not saved_paths:
            labels_n, label_columns = _labels_from_metadata_list(metadata_collected)
            print(f"Built {len(label_columns)} label columns from first run metadata")
        embeddings_dict = {
            "hsc_encoder1": e1,
            "hsc_encoder2": e2,
            "legacy_encoder1": e3,
            "legacy_encoder2": e4,
            "hsc_legacy_encoder1": e5,
            "hsc_legacy_encoder2": e6,
        }
        assert embeddings_dict["hsc_encoder1"].shape[0] == n_use
        save_downstream_h5(
            output_h5,
            embeddings_dict=embeddings_dict,
            labels_n=labels_n,
            label_columns=label_columns,
        )
        saved_paths.append(output_h5)

    print("Loading untrained model...")
    model_untrained = _load_untrained_model().to(device)
    print("Generating embeddings with untrained model...")
    e1, e2, e3, e4, e5, e6, _ = generate_embeddings(
        model_untrained, dataset, device, batch_size,
        shuffle=SHUFFLE_LOADER, seed=SEED if SHUFFLE_LOADER else None,
    )
    embeddings_untrained = {
        "hsc_encoder1": e1,
        "hsc_encoder2": e2,
        "legacy_encoder1": e3,
        "legacy_encoder2": e4,
        "hsc_legacy_encoder1": e5,
        "hsc_legacy_encoder2": e6,
    }
    save_downstream_h5(
        OUTPUT_DOWNSTREAM_UNTRAINED_H5,
        embeddings_dict=embeddings_untrained,
        labels_n=labels_n,
        label_columns=label_columns,
    )

    return saved_paths, OUTPUT_DOWNSTREAM_UNTRAINED_H5


def test_dataloader():
    """Quick test: shapes and one batch via DataLoader."""
    print("\n--- Testing NeighborsSimpleDataset ---")
    dataset = NeighborsSimpleDataset(hdf5_path=NEIGHBORS_HDF5)
    n = min(100, len(dataset))
    subset = Subset(dataset, range(n))
    hsc, leg, meta = subset[0]
    assert hsc.shape == leg.shape, f"Shape mismatch: hsc {hsc.shape} vs legacy {leg.shape}"
    assert hsc.dtype == torch.float32 and leg.dtype == torch.float32
    assert len(hsc.shape) == 3
    assert isinstance(meta, dict) and "idx" in meta
    print(f"  Single sample: hsc {hsc.shape}, legacy {leg.shape}, metadata keys: {len(meta)}")

    loader = DataLoader(
        subset,
        batch_size=4,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_neighbors_simple,
    )
    h_batch, l_batch, meta_list = next(iter(loader))
    assert h_batch.shape == (4, *hsc.shape)
    assert len(meta_list) == 4
    print(f"  Batch (size 4): hsc {h_batch.shape}, legacy {l_batch.shape}, metadata list len {len(meta_list)}")
    print("--- All dataloader tests passed ---\n")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(
        description="Neighbors downstream prep (NeighborsSimpleDataset + metadata from H5)"
    )
    p.add_argument(
        "mode",
        nargs="?",
        default="prepare",
        choices=("test", "prepare"),
        help="test=dataloader only; prepare=embeddings+metadata labels -> H5",
    )
    args = p.parse_args()
    if args.mode == "test":
        test_dataloader()
    else:
        prepare_downstream_dataset()
