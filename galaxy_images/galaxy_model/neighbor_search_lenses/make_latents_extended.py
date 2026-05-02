"""
Build an extended latent file covering ALL source types in neighbours_v2.h5:
  source_type=0: both HSC and Legacy images valid  (103k entries)
  source_type=1: HSC image only                    (262k entries)
  source_type=2: Legacy image only                 (101k entries)

Produces two separate embedding arrays in the output HDF5:
  hsc_index_mmu      (N_hsc,)    raw h5 rows for source_type ∈ {0,1}
  hsc_source_type    (N_hsc,)    0 or 1
  hsc_physics        (N_hsc, D)  encoder_1(hsc_image)
  legacy_index_mmu   (N_leg,)    raw h5 rows for source_type ∈ {0,2}
  legacy_source_type (N_leg,)    0 or 2
  legacy_physics     (N_leg, D)  encoder_1(legacy_image)

Run from galaxy_model/:
  python neighbor_search_lenses/make_latents_extended.py \
    --checkpoint outputs/neighbors_all_attn/2026-04-05/checkpoints/best-epoch=228-step=87000.ckpt \
    --output neighbor_search/neighbor_latents_extended.h5
"""
import sys
from pathlib import Path

_here = Path(__file__).resolve().parent
_src = _here.parent  # galaxy_model/
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

import argparse

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

NEIGHBORS_HDF5 = "/work1/jeroenaudenaert/pablomer/data/neighbours_v2.h5"
DEFAULT_CHECKPOINT = str(
    _src / "outputs/neighbors_all_attn/2026-04-05/checkpoints/best-epoch=228-step=87000.ckpt"
)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SingleSurveyDataset(Dataset):
    """Loads images for a given survey from a fixed list of raw h5 rows."""

    def __init__(self, h5_path: str, row_indices: np.ndarray, survey: str, crop_size: int = 48):
        self.h5_path = h5_path
        self.row_indices = row_indices
        self.survey = survey
        self.crop_size = crop_size
        self.file = None

    def _open(self):
        if self.file is None:
            self.file = h5py.File(self.h5_path, "r", libver="latest", swmr=True)

    def __len__(self):
        return len(self.row_indices)

    def __getitem__(self, i):
        from neighbors import preprocess_raw_image
        self._open()
        raw_row = int(self.row_indices[i])
        key = "images_hsc" if self.survey == "hsc" else "images_legacy"
        img = self.file[key][raw_row]
        img = preprocess_raw_image(img, self.survey, self.crop_size)[:4]
        return img, raw_row


def _collate(batch):
    imgs = torch.stack([b[0] for b in batch])
    rows = np.array([b[1] for b in batch], dtype=np.int64)
    return imgs, rows


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _load_model(checkpoint_path: str, device: torch.device):
    from double_train_fm_neighbors import ConditionalFlowMatchingModule
    model = ConditionalFlowMatchingModule.load_from_checkpoint(
        checkpoint_path, map_location="cpu"
    )
    model.eval()
    torch.set_grad_enabled(False)
    return model.to(device)


# ---------------------------------------------------------------------------
# Encoding + streaming HDF5 write
# ---------------------------------------------------------------------------

def encode_and_save(
    model,
    device: torch.device,
    h5_path: str,
    row_indices: np.ndarray,
    source_types_for_rows: np.ndarray,
    survey: str,
    out_file: h5py.File,
    prefix: str,        # "hsc" or "legacy"
    batch_size: int = 256,
):
    """Encode images for `row_indices` and stream-write into `out_file` under `prefix`."""
    n = len(row_indices)
    dataset = SingleSurveyDataset(h5_path, row_indices, survey)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=0, collate_fn=_collate)

    emb_ds = None
    written = 0
    with torch.no_grad():
        for imgs, rows in loader:
            phys = model.encoder_1(imgs.to(device)).cpu().flatten(start_dim=1).numpy().astype(np.float32)
            bs = phys.shape[0]

            if emb_ds is None:
                D = phys.shape[1]
                out_file.create_dataset(f"{prefix}_index_mmu",   shape=(n,),    dtype=np.int64)
                out_file.create_dataset(f"{prefix}_source_type", shape=(n,),    dtype=np.int8)
                out_file.create_dataset(f"{prefix}_physics",     shape=(n, D),  dtype=np.float32)
                emb_ds = True

            sl = slice(written, written + bs)
            out_file[f"{prefix}_index_mmu"][sl]   = rows
            out_file[f"{prefix}_source_type"][sl] = source_types_for_rows[written: written + bs]
            out_file[f"{prefix}_physics"][sl]     = phys
            written += bs

            if written % 10000 < batch_size or written >= n:
                print(f"    [{prefix}] {written}/{n}", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Build extended latent file for all source types (0, 1, 2)."
    )
    p.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    p.add_argument("--neighbors-h5", default=NEIGHBORS_HDF5)
    p.add_argument(
        "--output",
        type=Path,
        default=_src / "neighbor_search/neighbor_latents_extended.h5",
    )
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    try:
        torch.backends.cuda.preferred_blas_library("hipblas")
    except Exception:
        pass

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")

    # --- Partition rows by source type ---
    print(f"Reading source_type from {args.neighbors_h5} ...")
    with h5py.File(args.neighbors_h5, "r") as f:
        source_types = f["source_type"][:]

    hsc_mask = np.isin(source_types, [0, 1])
    leg_mask = np.isin(source_types, [0, 2])
    hsc_rows = np.where(hsc_mask)[0].astype(np.int64)
    leg_rows  = np.where(leg_mask)[0].astype(np.int64)
    hsc_st = source_types[hsc_rows]
    leg_st  = source_types[leg_rows]

    print(f"HSC entries (source_type 0+1): {len(hsc_rows):,}")
    print(f"Legacy entries (source_type 0+2): {len(leg_rows):,}")
    print(f"Combined space will be {len(hsc_rows) + len(leg_rows):,}")

    # --- Load model ---
    print(f"Loading model from {args.checkpoint} ...")
    model = _load_model(args.checkpoint, device)

    # --- Encode and write ---
    args.output.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing to {args.output} ...")
    with h5py.File(args.output, "w") as out:
        out.attrs["checkpoint"] = str(args.checkpoint)
        out.attrs["neighbors_h5"] = str(args.neighbors_h5)
        out.attrs["format"] = "extended"

        print("Encoding HSC images ...")
        encode_and_save(model, device, args.neighbors_h5,
                        hsc_rows, hsc_st, "hsc", out, "hsc", args.batch_size)

        print("Encoding Legacy images ...")
        encode_and_save(model, device, args.neighbors_h5,
                        leg_rows, leg_st, "legacy", out, "legacy", args.batch_size)

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    print(f"\nDone. Saved {args.output}")


if __name__ == "__main__":
    main()
