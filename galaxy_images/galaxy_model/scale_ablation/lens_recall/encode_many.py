#!/usr/bin/env python
"""Encode the gallery for MANY checkpoints, loading the image store only ONCE.

The whole point: the RAM48 store is 17 GB and takes minutes to read off Ceph. Running
encode_gallery.py ten times would pay that ten times. Here we build the dataset once and
loop the checkpoints over it, so the marginal cost per checkpoint is just the GPU pass.

  python encode_many.py --out-dir latents/ \
      --checkpoint scale-1000-75k:/path/a.ckpt \
      --checkpoint base-87k:/path/b.ckpt ...
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

_here = Path(__file__).resolve().parent
_gm = _here.parents[1]
_root = _gm.parents[1]
for p in (str(_gm), str(_root), str(_here)):
    if p not in sys.path:
        sys.path.insert(0, p)

from encode_gallery import (  # noqa: E402
    _H5Gallery, _Ram48Gallery, _collate, anchor_rows_and_mapping, check_not_degenerate,
    NEIGHBOURS_H5, RAM48_DIR,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", action="append", default=[], metavar="LABEL:PATH",
                    help="repeatable")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--source", choices=["h5", "ram48"], default="ram48")
    ap.add_argument("--neighbours-h5", default=NEIGHBOURS_H5)
    ap.add_argument("--ram48-dir", default=RAM48_DIR)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--num-workers", type=int, default=8)
    args = ap.parse_args()

    specs = []
    for s in args.checkpoint:
        label, path = s.split(":", 1)
        specs.append((label, Path(path)))
    missing = [f"{l} -> {p}" for l, p in specs if not p.exists()]
    if missing:
        raise SystemExit("missing checkpoints:\n  " + "\n  ".join(missing))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    todo = [(l, p) for l, p in specs if not (args.out_dir / f"{l}.h5").exists()]
    print(f"[many] {len(specs)} checkpoints, {len(todo)} to encode "
          f"({len(specs) - len(todo)} cached)", flush=True)
    if not todo:
        return

    # neighbours_v2 and the RAM48 catalog use DIFFERENT row orders (they agree on only 4.8%
    # of rows). Identity stays in neighbours_v2 space; cat_rows is only for fetching pixels.
    nb_rows, cat_rows, oid_all = anchor_rows_and_mapping(args.neighbours_h5, args.ram48_dir)
    rows = nb_rows
    print(f"[many] {len(rows):,} anchors | source={args.source}", flush=True)

    ds = (_H5Gallery(args.neighbours_h5, nb_rows) if args.source == "h5"
          else _Ram48Gallery(args.ram48_dir, cat_rows, nb_rows))
    nw = args.num_workers if args.source == "h5" else 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(False)

    from double_train_fm_neighbors import ConditionalFlowMatchingModule

    for n, (label, ckpt) in enumerate(todo, 1):
        out = args.out_dir / f"{label}.h5"
        t0 = time.time()
        model = ConditionalFlowMatchingModule.load_from_checkpoint(str(ckpt), map_location="cpu")
        model.eval().to(device)
        t_load = time.time() - t0

        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=nw, collate_fn=_collate, pin_memory=True)
        acc = {k: [] for k in ("hp", "hi", "lp", "li")}
        got = []
        t1 = time.time()
        for hsc, leg, r in loader:
            hsc, leg = hsc.to(device, non_blocking=True), leg.to(device, non_blocking=True)
            acc["hp"].append(model.encoder_1(hsc).flatten(1).cpu().numpy().astype(np.float32))
            acc["hi"].append(model.encoder_2(hsc).flatten(1).cpu().numpy().astype(np.float32))
            acc["lp"].append(model.encoder_1(leg).flatten(1).cpu().numpy().astype(np.float32))
            acc["li"].append(model.encoder_2(leg).flatten(1).cpu().numpy().astype(np.float32))
            got.append(r)
        cat = {k: np.concatenate(v, 0) for k, v in acc.items()}
        got = np.concatenate(got)
        check_not_degenerate(f"{label}/hsc_physics", cat["hp"])

        with h5py.File(out, "w") as f:
            f.create_dataset("index_mmu", data=got)
            f.create_dataset("object_id_hsc", data=np.array([oid_all[r].encode() for r in got]))
            f.create_dataset("hsc_physics", data=cat["hp"])
            f.create_dataset("hsc_instrument", data=cat["hi"])
            f.create_dataset("legacy_physics", data=cat["lp"])
            f.create_dataset("legacy_instrument", data=cat["li"])
            f.attrs["checkpoint"] = str(ckpt)
            f.attrs["label"] = label
            f.attrs["source"] = args.source
        enc = time.time() - t1
        print(f"[many] {n}/{len(todo)} {label:<22s} ckpt-load {t_load:5.1f}s  "
              f"encode {enc / 60:5.2f}m ({len(got) / enc:.0f} img/s)  -> {out.name}", flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
