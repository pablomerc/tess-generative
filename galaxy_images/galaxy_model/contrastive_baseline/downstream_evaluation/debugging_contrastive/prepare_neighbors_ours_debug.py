"""
Prepare neighbors-only downstream H5 for OURS model (trained-only embeddings).

Creates:
  - downstream_neighbors_ours_{suffix}.h5

This debug variant uses exactly 8k examples from neighbors with indices >= 70400.
"""

import argparse
import importlib.util
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader as TorchDataLoader, Subset

_here = Path(__file__).resolve().parent
_project_root = _here.parents[4]  # .../tess-generative
_galaxy_model_root = _project_root / "galaxy_images" / "galaxy_model"
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
if str(_galaxy_model_root) not in sys.path:
    sys.path.insert(0, str(_galaxy_model_root))

from galaxy_images.galaxy_model.neighbors import NeighborsSimpleDataset


NEIGHBORS_HDF5 = "/data/vision/billf/scratch/pablomer/data/neighbours_v2.h5"
NEIGHBORS_START_INDEX = 70400
NEIGHBORS_COUNT = 8000
NEIGHBORS_SEED = 42
SHUFFLE_NEIGHBORS = True

EMBEDDING_NAMES = (
    "hsc_encoder1",
    "hsc_encoder2",
    "legacy_encoder1",
    "legacy_encoder2",
    "hsc_legacy_encoder1",
    "hsc_legacy_encoder2",
)


def _load_model(checkpoint_path, module_filename):
    module_path = _galaxy_model_root / module_filename
    if not module_path.exists():
        raise FileNotFoundError(f"Module file not found: {module_path}")

    module_name = module_filename.replace(".py", "").replace(".", "_")
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    if not hasattr(mod, "ConditionalFlowMatchingModule"):
        raise AttributeError(f"ConditionalFlowMatchingModule not found in {module_path}")

    model = mod.ConditionalFlowMatchingModule.load_from_checkpoint(checkpoint_path, map_location="cpu")
    model.eval()
    torch.set_grad_enabled(False)
    return model


def collate_neighbors(batch):
    hsc = torch.stack([b[0] for b in batch])
    leg = torch.stack([b[1] for b in batch])
    meta = [b[2] for b in batch]
    return hsc, leg, meta


def _labels_from_metadata_list(metadata_list):
    if not metadata_list:
        return {}, []
    keys = list(metadata_list[0].keys())
    labels_n, valid_columns = {}, []
    for col in keys:
        try:
            vals = [m[col] for m in metadata_list]
            arr = np.array(vals)
            if arr.dtype.kind in "iuflb" and arr.shape[0] == len(metadata_list):
                labels_n[col] = arr.astype(np.float32) if arr.dtype.kind == "f" else arr
                valid_columns.append(col)
        except (TypeError, ValueError):
            continue
    return labels_n, valid_columns


def generate_embeddings_neighbors(model, dataset, device, batch_size=256, shuffle=False, seed=None):
    loader_kw = dict(batch_size=batch_size, num_workers=0, collate_fn=collate_neighbors)
    if shuffle:
        g = torch.Generator()
        if seed is not None:
            g.manual_seed(seed)
        loader_kw["shuffle"] = True
        loader_kw["generator"] = g
    else:
        loader_kw["shuffle"] = False

    loader = TorchDataLoader(dataset, **loader_kw)
    n_batches = len(loader)
    print(f"[trained] Embedding pass: {len(dataset)} samples, {n_batches} batches, batch_size={batch_size}")

    hsc_1, hsc_2, leg_1, leg_2 = [], [], [], []
    metadata_collected = []
    start_t = time.perf_counter()

    with torch.no_grad():
        for batch_idx, (hsc_im, leg_im, meta_list) in enumerate(loader, start=1):
            metadata_collected.extend(meta_list)
            h, l = hsc_im.to(device), leg_im.to(device)
            h1 = model.encoder_1(h)
            h2 = model.encoder_2(h)
            l1 = model.encoder_1(l)
            l2 = model.encoder_2(l)
            hsc_1.append(h1.cpu())
            hsc_2.append(h2.cpu())
            leg_1.append(l1.cpu())
            leg_2.append(l2.cpu())
            if batch_idx == 1 or batch_idx % 10 == 0 or batch_idx == n_batches:
                elapsed = time.perf_counter() - start_t
                print(
                    f"[trained] Batch {batch_idx}/{n_batches} "
                    f"({len(metadata_collected)}/{len(dataset)} samples, {elapsed:.1f}s elapsed)"
                )

    e1 = torch.cat(hsc_1, dim=0).flatten(start_dim=1)
    e2 = torch.cat(hsc_2, dim=0).flatten(start_dim=1)
    e3 = torch.cat(leg_1, dim=0).flatten(start_dim=1)
    e4 = torch.cat(leg_2, dim=0).flatten(start_dim=1)
    e5 = torch.cat([torch.cat([h.flatten(1), l.flatten(1)], dim=1) for h, l in zip(hsc_1, leg_1)], dim=0)
    e6 = torch.cat([torch.cat([h.flatten(1), l.flatten(1)], dim=1) for h, l in zip(hsc_2, leg_2)], dim=0)
    return (e1.numpy(), e2.numpy(), e3.numpy(), e4.numpy(), e5.numpy(), e6.numpy()), metadata_collected


def save_downstream_h5(path, real_dict, labels_n, label_columns):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        for name in EMBEDDING_NAMES:
            f.create_dataset(name, data=real_dict[name], compression="gzip", compression_opts=4)
        for col in label_columns:
            f.create_dataset(f"labels/{col}", data=labels_n[col], compression="gzip", compression_opts=4)
        f.attrs["embedding_names"] = list(EMBEDDING_NAMES)
        f.attrs["num_examples"] = real_dict[EMBEDDING_NAMES[0]].shape[0]
        f.attrs["label_columns"] = list(label_columns)
    print(f"Saved: {path}")


def prepare_neighbors(checkpoint_path, module_filename, output_dir, batch_size, suffix):
    print(f"[Prepare-Ours] Loading neighbors dataset from: {NEIGHBORS_HDF5}")
    full_dataset = NeighborsSimpleDataset(hdf5_path=NEIGHBORS_HDF5)
    end_index = NEIGHBORS_START_INDEX + NEIGHBORS_COUNT
    if len(full_dataset) < end_index:
        raise ValueError(
            f"Neighbors dataset has {len(full_dataset)} entries, but this run requires at least {end_index}."
        )

    dataset = Subset(full_dataset, range(NEIGHBORS_START_INDEX, end_index))
    print(
        "[Prepare-Ours] Slice configuration: "
        f"start={NEIGHBORS_START_INDEX}, count={NEIGHBORS_COUNT}, end_exclusive={end_index}"
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[Prepare-Ours] Using device: {device}")
    print(f"[Prepare-Ours] Checkpoint: {checkpoint_path}")
    print(f"[Prepare-Ours] Module: {module_filename}")

    print("[Prepare-Ours] Loading trained model...")
    model = _load_model(checkpoint_path, module_filename).to(device)
    print("[Prepare-Ours] Running embedding extraction...")
    e_arrays, metadata_collected = generate_embeddings_neighbors(
        model, dataset, device, batch_size, shuffle=SHUFFLE_NEIGHBORS, seed=NEIGHBORS_SEED
    )
    real = dict(zip(EMBEDDING_NAMES, e_arrays))
    labels_n, label_columns = _labels_from_metadata_list(metadata_collected)
    print(f"[Prepare-Ours] Collected {len(label_columns)} numeric label columns.")

    out_path = output_dir / f"downstream_neighbors_ours_{suffix}.h5"
    print(f"[Prepare-Ours] Writing downstream H5 to: {out_path}")
    save_downstream_h5(out_path, real, labels_n, label_columns)
    return out_path


def main():
    p = argparse.ArgumentParser(description="Prepare neighbors-only debug embeddings for OURS model (trained only).")
    p.add_argument("--checkpoint", required=True, help="Path to OURS checkpoint (.ckpt)")
    p.add_argument("--module", default="double_train_fm_neighbors.py", help="Model-def module under galaxy_model")
    p.add_argument("--suffix", default=None, help="Output suffix. Default: checkpoint stem")
    p.add_argument("--output-dir", type=Path, default=_here / "outputs", help="Output directory for H5 file")
    p.add_argument("--batch-size", type=int, default=256)
    args = p.parse_args()

    suffix = args.suffix or Path(args.checkpoint).stem
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[Neighbors Ours Debug]")
    saved = prepare_neighbors(args.checkpoint, args.module, output_dir, args.batch_size, suffix)

    print("\nDone. Saved:")
    print(f"  {saved}")


if __name__ == "__main__":
    main()
