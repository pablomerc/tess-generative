"""
Prepare neighbors-only downstream H5 for the contrastive dual-encoder baseline.

Creates:
  - downstream_neighbors_{suffix}.h5

This debug variant uses exactly 8k examples from the neighbors dataset,
restricted to indices >= 70400.
"""

import argparse
import inspect
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader as TorchDataLoader, Subset

_here = Path(__file__).resolve().parent
# .../tess-generative/galaxy_images/galaxy_model/contrastive_baseline/downstream_evaluation/debugging_contrastive
#                                    ^ project root is parents[4] (.../tess-generative)
_project_root = _here.parents[4]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from galaxy_images.galaxy_model.contrastive_baseline.dual_encoder_contrastive import (
    DualEncoderContrastiveModule,
)
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


def _load_model(checkpoint_path):
    model = DualEncoderContrastiveModule.load_from_checkpoint(checkpoint_path, map_location="cpu")
    model.eval()
    torch.set_grad_enabled(False)
    return model


def _load_untrained_model(checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    hparams = ckpt.get("hyper_parameters", {})
    valid_keys = set(inspect.signature(DualEncoderContrastiveModule.__init__).parameters.keys())
    init_kwargs = {k: v for k, v in hparams.items() if k in valid_keys}
    model = DualEncoderContrastiveModule(**init_kwargs)
    model.eval()
    torch.set_grad_enabled(False)
    return model


def _encode_hsc_legacy(model, hsc_batch, legacy_batch):
    h_g = model.encoder_galaxy(hsc_batch)
    h_i = model.encoder_instrument(hsc_batch)
    l_g = model.encoder_galaxy(legacy_batch)
    l_i = model.encoder_instrument(legacy_batch)
    return h_g, h_i, l_g, l_i


def generate_embeddings_neighbors(model, dataset, device, batch_size=256, shuffle=False, seed=None, stage_label="model"):
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

    hsc_1, hsc_2, leg_1, leg_2 = [], [], [], []
    metadata_collected = []
    n_batches = len(loader)
    print(f"[{stage_label}] Embedding pass: {len(dataset)} samples, {n_batches} batches, batch_size={batch_size}")
    start_t = time.perf_counter()
    with torch.no_grad():
        for batch_idx, (hsc_im, leg_im, meta_list) in enumerate(loader, start=1):
            metadata_collected.extend(meta_list)
            h, l = hsc_im.to(device), leg_im.to(device)
            e1, e2, e3, e4 = _encode_hsc_legacy(model, h, l)
            hsc_1.append(e1.cpu())
            hsc_2.append(e2.cpu())
            leg_1.append(e3.cpu())
            leg_2.append(e4.cpu())
            if batch_idx == 1 or batch_idx % 10 == 0 or batch_idx == n_batches:
                elapsed = time.perf_counter() - start_t
                print(
                    f"[{stage_label}] Batch {batch_idx}/{n_batches} "
                    f"({len(metadata_collected)}/{len(dataset)} samples, {elapsed:.1f}s elapsed)"
                )

    e1 = torch.cat(hsc_1, dim=0).flatten(start_dim=1)
    e2 = torch.cat(hsc_2, dim=0).flatten(start_dim=1)
    e3 = torch.cat(leg_1, dim=0).flatten(start_dim=1)
    e4 = torch.cat(leg_2, dim=0).flatten(start_dim=1)
    e5 = torch.cat([torch.cat([h, l], dim=1) for h, l in zip(hsc_1, leg_1)], dim=0)
    e6 = torch.cat([torch.cat([h, l], dim=1) for h, l in zip(hsc_2, leg_2)], dim=0)
    return (e1.numpy(), e2.numpy(), e3.numpy(), e4.numpy(), e5.numpy(), e6.numpy()), metadata_collected


def make_random_embeddings_like(real_embeddings_dict, seed=42):
    rng = np.random.default_rng(seed)
    out = {}
    for k, v in real_embeddings_dict.items():
        out[k] = rng.standard_normal(v.shape, dtype=np.float32)
    return out


def save_downstream_h5_all_variants(path, real_dict, untrained_dict, random_dict, labels_n, label_columns):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        for name in EMBEDDING_NAMES:
            f.create_dataset(name, data=real_dict[name], compression="gzip", compression_opts=4)
            f.create_dataset(name + "_untrained", data=untrained_dict[name], compression="gzip", compression_opts=4)
            f.create_dataset(name + "_random", data=random_dict[name], compression="gzip", compression_opts=4)
        for col in label_columns:
            f.create_dataset(f"labels/{col}", data=labels_n[col], compression="gzip", compression_opts=4)
        f.attrs["embedding_names"] = list(EMBEDDING_NAMES)
        f.attrs["num_examples"] = real_dict[EMBEDDING_NAMES[0]].shape[0]
        f.attrs["label_columns"] = list(label_columns)
    print(f"Saved: {path}")


def _run_model_variants(get_real_emb_fn, checkpoint_path, seed):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[Prepare] Using device: {device}")
    print(f"[Prepare] Checkpoint: {checkpoint_path}")

    print("[Prepare] Loading trained checkpoint model...")
    model = _load_model(checkpoint_path).to(device)
    print("[Prepare] Running embeddings with trained model...")
    real = get_real_emb_fn(model, stage_label="trained")
    print("[Prepare] Trained-model embeddings complete.")
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    print("[Prepare] Loading untrained model from checkpoint hyperparameters...")
    model_untrained = _load_untrained_model(checkpoint_path).to(device)
    print("[Prepare] Running embeddings with untrained model...")
    untrained = get_real_emb_fn(model_untrained, stage_label="untrained")
    print("[Prepare] Untrained-model embeddings complete.")
    del model_untrained
    if device.type == "cuda":
        torch.cuda.empty_cache()

    print("[Prepare] Generating random baseline embeddings...")
    random_emb = make_random_embeddings_like(real, seed=seed)
    print("[Prepare] Random baseline embeddings complete.")
    return real, untrained, random_emb


def prepare_neighbors(checkpoint_path, output_dir, batch_size, seed, suffix):
    print(f"[Neighbors Debug] Loading neighbors dataset from: {NEIGHBORS_HDF5}")
    full_dataset = NeighborsSimpleDataset(hdf5_path=NEIGHBORS_HDF5)
    end_index = NEIGHBORS_START_INDEX + NEIGHBORS_COUNT
    if len(full_dataset) < end_index:
        raise ValueError(
            f"Neighbors dataset has {len(full_dataset)} entries, but this debug run requires at least {end_index}."
        )

    dataset = Subset(full_dataset, range(NEIGHBORS_START_INDEX, end_index))
    print(
        "[Neighbors Debug] Slice configuration: "
        f"start={NEIGHBORS_START_INDEX}, count={NEIGHBORS_COUNT}, end_exclusive={end_index}"
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    meta_cache = {"labels_n": None, "label_columns": None}

    def _emb_fn(model, stage_label):
        e_arrays, metadata_collected = generate_embeddings_neighbors(
            model,
            dataset,
            device,
            batch_size,
            shuffle=SHUFFLE_NEIGHBORS,
            seed=NEIGHBORS_SEED,
            stage_label=stage_label,
        )
        if meta_cache["labels_n"] is None:
            labels_n, label_columns = _labels_from_metadata_list(metadata_collected)
            meta_cache["labels_n"] = labels_n
            meta_cache["label_columns"] = label_columns
            print(f"[Neighbors Debug] Collected {len(label_columns)} numeric label columns.")
        return dict(zip(EMBEDDING_NAMES, e_arrays))

    real, untrained, random_emb = _run_model_variants(_emb_fn, checkpoint_path, seed)
    out_path = output_dir / f"downstream_neighbors_{suffix}.h5"
    print(f"[Prepare] Writing downstream H5 to: {out_path}")
    save_downstream_h5_all_variants(
        out_path, real, untrained, random_emb, meta_cache["labels_n"], meta_cache["label_columns"]
    )
    return out_path


def main():
    p = argparse.ArgumentParser(description="Prepare neighbors-only debug embeddings for contrastive baseline.")
    p.add_argument("--checkpoint", required=True, help="Path to contrastive baseline checkpoint (.ckpt)")
    p.add_argument("--suffix", default=None, help="Output suffix. Default: checkpoint stem")
    p.add_argument("--output-dir", type=Path, default=_here, help="Output directory for H5 files")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    suffix = args.suffix or Path(args.checkpoint).stem
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[Neighbors Debug]")
    saved = prepare_neighbors(args.checkpoint, output_dir, args.batch_size, args.seed, suffix)

    print("\nDone. Saved:")
    print(f"  {saved}")


if __name__ == "__main__":
    main()
