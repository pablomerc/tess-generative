#!/usr/bin/env python
"""Encode the 103,741-anchor gallery into physics + instrument latents for kNN retrieval.

*** THE ROW-SPACE TRAP (this cost a wasted GPU-hour; do not re-introduce it) ***
`neighbours_v2.h5` and `efficient_neighs_48/` (the RAM48 store + catalog.parquet) BOTH hold
468,197 records, but in COMPLETELY DIFFERENT ORDER -- they agree on `object_id_hsc` for only
4.8% of rows, because the efficient store is Hilbert-sorted on (RA, Dec). Indexing one with
the other's row numbers silently returns the wrong galaxy, and often a BLANK one: `hsc48.bin`
is a constant fill for Legacy-only rows (source_type==2) and vice versa. Constant input makes
the encoders emit near-identical vectors, so kNN degenerates to returning consecutive row
indices at identical distance -- which looks plausible until you read it closely.

So: identity is carried in **neighbours_v2 row space** throughout (that is what the recorded
Fig-5 neighbours and the display stamps use), and the RAM48 path maps into catalog row space
only to fetch pixels, via an object_id_hsc join.

Sources:
  --source h5      raw 160x160 from neighbours_v2.h5, preprocessed live. Faithful to the
                   paper pipeline; ~96 GB of reads per pass.
  --source ram48   the pre-baked 48x48 float16 store training used. 17 GB, loads in seconds,
                   then encoding is pure GPU (~11k img/s). Differs only by fp16 rounding.

Speed note: deliberately avoids NeighborsSimpleDataset, which reads ~60 metadata scalars per
item (~6.2M extra random HDF5 reads over a full pass) and is unusably I/O-bound here.

Output HDF5 (index_mmu is ALWAYS a neighbours_v2.h5 row):
  index_mmu, object_id_hsc, hsc_physics, hsc_instrument, legacy_physics, legacy_instrument
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

_here = Path(__file__).resolve().parent
_gm = _here.parents[1]
_root = _gm.parents[1]
for p in (str(_gm), str(_root)):
    if p not in sys.path:
        sys.path.insert(0, p)

NEIGHBOURS_H5 = "/orcd/pool/007/pablomer/neighbours_v2.h5"
RAM48_DIR = "/orcd/pool/007/pablomer/efficient_neighs_48"
N_ROWS_TOTAL = 468197


def anchor_rows_and_mapping(neighbours_h5: str, ram48_dir: str):
    """Return (nb_rows, cat_rows, oid_all) for source_type==0 anchors.

    nb_rows  : rows into neighbours_v2.h5  (the identity space)
    cat_rows : matching rows into the RAM48 store / catalog.parquet
    """
    import pandas as pd
    with h5py.File(neighbours_h5, "r") as f:
        st = f["source_type"][:]
        oid_all = f["object_id_hsc"][:]
    oid_all = np.array([o.decode() if hasattr(o, "decode") else str(o) for o in oid_all])
    nb_rows = np.where(st == 0)[0]

    cat = pd.read_parquet(Path(ram48_dir) / "catalog.parquet",
                          columns=["object_id_hsc", "source_type"])
    pos = {o: i for i, o in enumerate(cat["object_id_hsc"].values) if o}
    cat_rows = np.array([pos.get(oid_all[r], -1) for r in nb_rows], dtype=np.int64)

    bad = int((cat_rows < 0).sum())
    if bad:
        raise RuntimeError(f"{bad} anchors could not be mapped into the RAM48 catalog")
    if not (cat["source_type"].values[cat_rows] == 0).all():
        raise RuntimeError("mapped rows are not all anchors in the catalog - join is wrong")
    return nb_rows, cat_rows, oid_all


class _H5Gallery(Dataset):
    """Raw-stamp path. Indexed by neighbours_v2 rows."""

    def __init__(self, h5_path, nb_rows, crop_size: int = 48):
        self.h5_path, self.rows, self.crop_size = h5_path, nb_rows, crop_size
        self.f = None

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        from neighbors import preprocess_raw_image
        if self.f is None:
            self.f = h5py.File(self.h5_path, "r", libver="latest", swmr=True)
        row = int(self.rows[i])
        hsc = preprocess_raw_image(self.f["images_hsc"][row][:4], "hsc", self.crop_size)
        leg = preprocess_raw_image(self.f["images_legacy"][row][:4], "legacy", self.crop_size)
        return hsc[:4], leg[:4], row


class _Ram48Gallery(Dataset):
    """Pre-baked path. Pixels come from cat_rows; the row we REPORT is the nb_row."""

    def __init__(self, data_dir, cat_rows, nb_rows):
        d = Path(data_dir)
        shp = (N_ROWS_TOTAL, 4, 48, 48)
        print(f"[gallery] loading RAM48 store ({2*np.prod(shp)*2/1e9:.1f} GB) ...", flush=True)
        t0 = time.time()
        self.hsc = np.fromfile(d / "hsc48.bin", dtype=np.float16).reshape(shp)
        self.leg = np.fromfile(d / "legacy48.bin", dtype=np.float16).reshape(shp)
        print(f"[gallery] loaded in {time.time()-t0:.0f}s", flush=True)
        self.cat_rows, self.nb_rows = cat_rows, nb_rows

        # Guard: anchors must have real pixels in BOTH bands. Constant frames mean the row
        # space is wrong (see module docstring).
        blank = 0
        for i in np.linspace(0, len(cat_rows) - 1, 64).astype(int):
            c = int(cat_rows[i])
            if np.asarray(self.hsc[c], np.float32).std() == 0 or \
               np.asarray(self.leg[c], np.float32).std() == 0:
                blank += 1
        if blank:
            raise RuntimeError(
                f"{blank}/64 sampled anchors are CONSTANT frames in the RAM48 store - the row "
                f"mapping is wrong. Do not proceed; see the docstring.")
        print("[gallery] row-mapping guard passed (64/64 anchors have real pixels)", flush=True)

    def __len__(self):
        return len(self.cat_rows)

    def __getitem__(self, i):
        c = int(self.cat_rows[i])
        return (torch.from_numpy(self.hsc[c].astype(np.float32)),
                torch.from_numpy(self.leg[c].astype(np.float32)),
                int(self.nb_rows[i]))


def _collate(batch):
    return (torch.stack([b[0] for b in batch]),
            torch.stack([b[1] for b in batch]),
            np.array([b[2] for b in batch], dtype=np.int64))


def check_not_degenerate(name: str, emb: np.ndarray) -> None:
    """Catch the failure mode that produced consecutive-row kNN at identical distance."""
    spread = float(emb.std(axis=0).mean())
    if spread < 1e-3:
        raise RuntimeError(
            f"{name}: embeddings are degenerate (mean per-dim std {spread:.2e}). The encoder "
            f"is seeing constant input - check the row mapping.")
    print(f"[check] {name}: mean per-dim std {spread:.4f}  OK", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--source", choices=["h5", "ram48"], default="ram48")
    ap.add_argument("--neighbours-h5", default=NEIGHBOURS_H5)
    ap.add_argument("--ram48-dir", default=RAM48_DIR)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--num-workers", type=int, default=8)
    args = ap.parse_args()

    if args.out.exists():
        print(f"[gallery] {args.out} exists - skipping")
        return

    nb_rows, cat_rows, oid_all = anchor_rows_and_mapping(args.neighbours_h5, args.ram48_dir)
    print(f"[gallery] {len(nb_rows):,} anchors | source={args.source}", flush=True)

    from double_train_fm_neighbors import ConditionalFlowMatchingModule
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConditionalFlowMatchingModule.load_from_checkpoint(args.checkpoint, map_location="cpu")
    model.eval().to(device)
    torch.set_grad_enabled(False)

    ds = (_H5Gallery(args.neighbours_h5, nb_rows) if args.source == "h5"
          else _Ram48Gallery(args.ram48_dir, cat_rows, nb_rows))
    nw = args.num_workers if args.source == "h5" else 0
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=nw,
                        collate_fn=_collate, pin_memory=True)

    acc = {k: [] for k in ("hp", "hi", "lp", "li")}
    got = []
    t0 = time.time()
    for hsc, leg, r in loader:
        hsc, leg = hsc.to(device, non_blocking=True), leg.to(device, non_blocking=True)
        acc["hp"].append(model.encoder_1(hsc).flatten(1).cpu().numpy().astype(np.float32))
        acc["hi"].append(model.encoder_2(hsc).flatten(1).cpu().numpy().astype(np.float32))
        acc["lp"].append(model.encoder_1(leg).flatten(1).cpu().numpy().astype(np.float32))
        acc["li"].append(model.encoder_2(leg).flatten(1).cpu().numpy().astype(np.float32))
        got.append(r)
    emb = {k: np.concatenate(v, 0) for k, v in acc.items()}
    got = np.concatenate(got)
    check_not_degenerate("hsc_physics", emb["hp"])
    check_not_degenerate("hsc_instrument", emb["hi"])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.out, "w") as f:
        f.create_dataset("index_mmu", data=got)
        f.create_dataset("object_id_hsc", data=np.array([oid_all[r].encode() for r in got]))
        f.create_dataset("hsc_physics", data=emb["hp"])
        f.create_dataset("hsc_instrument", data=emb["hi"])
        f.create_dataset("legacy_physics", data=emb["lp"])
        f.create_dataset("legacy_instrument", data=emb["li"])
        f.attrs["checkpoint"] = str(args.checkpoint)
        f.attrs["source"] = args.source
    print(f"[gallery] wrote {args.out} in {(time.time()-t0)/60:.1f}m", flush=True)


if __name__ == "__main__":
    main()
