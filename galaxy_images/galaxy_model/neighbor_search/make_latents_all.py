"""
Build a large HDF5 of physics and instrument embeddings for the neighbors dataset,
for fast kNN search in latent space.

Uses the same pretrained model and loading convention as
downstream_evaluation/final/prepare_all.py. Loads NeighborsSimpleDataset for the
first N examples (default 200k), runs encoder_1 (physics) and encoder_2 (instrument)
on both HSC and Legacy images, and saves:
  - index_mmu: row index in the original neighbours HDF5 (for loading images later)
  - idx: position in this subset (0 .. N-1)
  - physics_embedding: encoder_1(hsc), shape (N, D1)
  - instrument_embedding: encoder_2(hsc), shape (N, D2)
  - legacy_physics_embedding: encoder_1(legacy), shape (N, D1)
  - legacy_instrument_embedding: encoder_2(legacy), shape (N, D2)

Run from galaxy_model/:
  python neighbor_search/make_latents_all.py --checkpoint PATH --module double_train_fm_neighbors.py
"""
import sys
from pathlib import Path

_here = Path(__file__).resolve().parent
# galaxy_model (contains neighbors.py, double_train_fm_*.py)
_src = _here.parent
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

import argparse
import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader as TorchDataLoader, Subset

# Same paths as prepare_all.py
NEIGHBORS_HDF5 = "/data/vision/billf/scratch/pablomer/data/neighbours_v2.h5"
DEFAULT_MAX_EXAMPLES = 200_000
DEFAULT_MODULE = "double_train_fm_neighbors.py"


def _load_model(checkpoint_path, module_filename):
    """Load double-encoder model from checkpoint (same as prepare_all.py)."""
    import importlib.util
    module_name = module_filename.replace(".py", "").replace(".", "_")
    spec = importlib.util.spec_from_file_location(module_name, _src / module_filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    model = mod.ConditionalFlowMatchingModule.load_from_checkpoint(
        checkpoint_path, map_location="cpu"
    )
    model.eval()
    torch.set_grad_enabled(False)
    return model


def collate_neighbors(batch):
    hsc = torch.stack([b[0] for b in batch])
    leg = torch.stack([b[1] for b in batch])
    meta = [b[2] for b in batch]
    return hsc, leg, meta


def generate_and_save_embeddings(model, dataset, device, output_path, n_use, checkpoint, module_filename, batch_size=256):
    """Stream-write physics + instrument embeddings to HDF5 batch by batch to avoid OOM."""
    loader = TorchDataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_neighbors,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    hdf5_file = None
    row = 0
    with torch.no_grad():
        for hsc_im, leg_im, meta_list in loader:
            h = hsc_im.to(device)
            l = leg_im.to(device)
            hsc_phys = model.encoder_1(h).cpu().flatten(start_dim=1).numpy().astype(np.float32)
            hsc_inst  = model.encoder_2(h).cpu().flatten(start_dim=1).numpy().astype(np.float32)
            leg_phys  = model.encoder_1(l).cpu().flatten(start_dim=1).numpy().astype(np.float32)
            leg_inst  = model.encoder_2(l).cpu().flatten(start_dim=1).numpy().astype(np.float32)
            index_mmu_batch = np.array([m["index_mmu"] for m in meta_list], dtype=np.int64)
            bs = hsc_phys.shape[0]

            if hdf5_file is None:
                # Create datasets on first batch now that we know embedding dims
                d_phys = hsc_phys.shape[1]
                d_inst = hsc_inst.shape[1]
                hdf5_file = h5py.File(output_path, "w")
                hdf5_file.create_dataset("idx",                        shape=(n_use,),         dtype=np.int64)
                hdf5_file.create_dataset("index_mmu",                  shape=(n_use,),         dtype=np.int64)
                hdf5_file.create_dataset("physics_embedding",          shape=(n_use, d_phys),  dtype=np.float32)
                hdf5_file.create_dataset("instrument_embedding",       shape=(n_use, d_inst),  dtype=np.float32)
                hdf5_file.create_dataset("legacy_physics_embedding",   shape=(n_use, d_phys),  dtype=np.float32)
                hdf5_file.create_dataset("legacy_instrument_embedding",shape=(n_use, d_inst),  dtype=np.float32)
                hdf5_file.attrs["num_examples"] = n_use
                hdf5_file.attrs["checkpoint"]   = str(checkpoint)
                hdf5_file.attrs["module"]        = module_filename

            end = row + bs
            hdf5_file["idx"][row:end]                         = np.arange(row, end, dtype=np.int64)
            hdf5_file["index_mmu"][row:end]                   = index_mmu_batch
            hdf5_file["physics_embedding"][row:end]           = hsc_phys
            hdf5_file["instrument_embedding"][row:end]        = hsc_inst
            hdf5_file["legacy_physics_embedding"][row:end]    = leg_phys
            hdf5_file["legacy_instrument_embedding"][row:end] = leg_inst
            row = end

            if row % 10000 == 0 or row >= n_use:
                print(f"  {row}/{n_use} encoded", flush=True)

    if hdf5_file is not None:
        hdf5_file.close()


def main():
    p = argparse.ArgumentParser(
        description="Build neighbor latent file (physics + instrument embeddings) for kNN search."
    )
    p.add_argument(
        "--checkpoint",
        required=True,
        help="Path to model checkpoint (.ckpt), same as prepare_all.py",
    )
    p.add_argument(
        "--module",
        default=DEFAULT_MODULE,
        help=f"Module filename (default: {DEFAULT_MODULE})",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output HDF5 path (default: neighbor_search/neighbor_latents_<suffix>.h5)",
    )
    p.add_argument(
        "--max-examples",
        type=int,
        default=DEFAULT_MAX_EXAMPLES,
        help=f"Max examples from neighbors dataset (default: {DEFAULT_MAX_EXAMPLES})",
    )
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--suffix", default=None, help="Suffix for output filename (default: from checkpoint dir)")
    p.add_argument("--neighbors-h5", default=NEIGHBORS_HDF5, help=f"Path to neighbours_v2.h5 (default: {NEIGHBORS_HDF5})")
    args = p.parse_args()

    from neighbors import NeighborsSimpleDataset

    full_dataset = NeighborsSimpleDataset(hdf5_path=args.neighbors_h5)
    n_total = len(full_dataset)
    n_use = min(args.max_examples, n_total)
    dataset = Subset(full_dataset, range(n_use))
    print(f"Neighbors dataset: using first {n_use} of {n_total} examples")

    if args.suffix is None:
        args.suffix = Path(args.checkpoint).parent.parent.name
    if args.output is None:
        args.output = _here / f"neighbor_latents_{args.suffix}.h5"
    args.output = Path(args.output)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = _load_model(args.checkpoint, args.module).to(device)
    print("Generating physics (encoder_1) and instrument (encoder_2) embeddings on HSC and Legacy...")
    print("Writing directly to HDF5 to avoid OOM ...")
    generate_and_save_embeddings(
        model, dataset, device,
        output_path=args.output,
        n_use=n_use,
        checkpoint=args.checkpoint,
        module_filename=args.module,
        batch_size=args.batch_size,
    )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    print(f"Saved: {args.output} (idx, index_mmu, physics/instrument/legacy_physics/legacy_instrument embeddings)")


if __name__ == "__main__":
    main()
