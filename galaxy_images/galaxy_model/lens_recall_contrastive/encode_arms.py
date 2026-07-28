#!/usr/bin/env python
"""Encode the retrieval gallery for arbitrary model families (FM *and* contrastive).

This is `scale_ablation/lens_recall/encode_many.py` generalised on one axis only: the model
class. That script hardcodes `ConditionalFlowMatchingModule` and `encoder_1`/`encoder_2`, so it
cannot load the contrastive dual encoder, whose branches are `encoder_galaxy`/`encoder_instrument`.
Everything else -- the anchor set, the row-space mapping, the RAM48 store, the degeneracy guards --
is IMPORTED from that module rather than reimplemented, so the two paths cannot drift.

Encoder convention is copied from `downstream_evaluation/engaging/prepare_combined.py:164-177`,
which is what produced the downstream R^2 tables:

    contrastive : e1 = encoder_galaxy(x)      e2 = encoder_instrument(x)
    FM / base   : e1 = encoder_1(x)           e2 = encoder_2(x)

so the deck and the R^2 table describe the *same vectors* (raw encoders, never the projection
heads).

The store is read ONCE and every arm is looped over it -- the whole reason this is a separate
script from encode_gallery.py.

    python encode_arms.py --out-dir latents \
      --arm 'label:relative/module.py:ClassName:/abs/path.ckpt' ...
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

_here = Path(__file__).resolve().parent
_gm = _here.parent                       # .../galaxy_images/galaxy_model
_root = _gm.parents[1]                   # .../tess-generative
_lr = _gm / "scale_ablation" / "lens_recall"
for p in (str(_lr), str(_gm), str(_root)):
    if p not in sys.path:
        sys.path.insert(0, p)

from encode_gallery import (  # noqa: E402  (from scale_ablation/lens_recall)
    _H5Gallery, _Ram48Gallery, _collate, anchor_rows_and_mapping, check_not_degenerate,
    NEIGHBOURS_H5, RAM48_DIR,
)

EXPECTED_WIDTH = 64   # both families flatten to 64: FM (4,16); contrastive conv1x1 (16,2,2)


def _load_class(module_relpath: str, class_name: str):
    """Import `class_name` from a .py file given relative to galaxy_model/."""
    path = _gm / module_relpath
    if not path.exists():
        raise SystemExit(f"module not found: {path}")
    mod_name = module_relpath.replace("/", "_").replace(".py", "")
    spec = importlib.util.spec_from_file_location(mod_name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, class_name):
        raise SystemExit(f"{class_name} not in {module_relpath}; have "
                         f"{[n for n in dir(mod) if not n.startswith('_')]}")
    return getattr(mod, class_name)


def _is_contrastive(model) -> bool:
    # Mirrors prepare_combined.py:164 exactly.
    return hasattr(model, "encoder_galaxy") and hasattr(model, "encoder_instrument")


def _branches(model):
    """Return (physics_branch, instrument_branch) callables for either family."""
    if _is_contrastive(model):
        return model.encoder_galaxy, model.encoder_instrument
    if hasattr(model, "encoder_1") and hasattr(model, "encoder_2"):
        return model.encoder_1, model.encoder_2
    raise SystemExit(f"{type(model).__name__}: no (encoder_galaxy, encoder_instrument) or "
                     f"(encoder_1, encoder_2) pair -- unsupported architecture for this deck")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", default=[], metavar="LABEL:MODULE:CLASS:CKPT",
                    help="repeatable. MODULE is relative to galaxy_model/.")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--source", choices=["h5", "ram48"], default="ram48")
    ap.add_argument("--neighbours-h5", default=NEIGHBOURS_H5)
    ap.add_argument("--ram48-dir", default=RAM48_DIR)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--expect-width", type=int, default=EXPECTED_WIDTH,
                    help="assert every arm's flattened latent width; 0 disables. Guards against "
                         "silently mixing an 'avg'-pooled arm into a conv1x1 comparison.")
    args = ap.parse_args()

    specs = []
    for s in args.arm:
        parts = s.split(":")
        if len(parts) < 4:
            raise SystemExit(f"--arm needs LABEL:MODULE:CLASS:CKPT, got {s!r}")
        label, module, cls_name = parts[0], parts[1], parts[2]
        ckpt = Path(":".join(parts[3:]))          # tolerate ':' in paths
        specs.append((label, module, cls_name, ckpt))
    missing = [f"{l} -> {c}" for l, _, _, c in specs if not c.exists()]
    if missing:
        raise SystemExit("missing checkpoints:\n  " + "\n  ".join(missing))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    todo = [s for s in specs if not (args.out_dir / f"{s[0]}.h5").exists()]
    print(f"[arms] {len(specs)} arms, {len(todo)} to encode "
          f"({len(specs) - len(todo)} already cached)", flush=True)
    if not todo:
        return

    # THE ROW-SPACE TRAP: identity stays in neighbours_v2 row space; cat_rows only fetches
    # pixels. See encode_gallery.py's module docstring -- the two stores agree on 4.8% of rows.
    nb_rows, cat_rows, oid_all = anchor_rows_and_mapping(args.neighbours_h5, args.ram48_dir)
    print(f"[arms] {len(nb_rows):,} anchors | source={args.source}", flush=True)

    ds = (_H5Gallery(args.neighbours_h5, nb_rows) if args.source == "h5"
          else _Ram48Gallery(args.ram48_dir, cat_rows, nb_rows))
    nw = args.num_workers if args.source == "h5" else 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(False)

    for n, (label, module, cls_name, ckpt) in enumerate(todo, 1):
        out = args.out_dir / f"{label}.h5"
        t0 = time.time()
        cls = _load_class(module, cls_name)
        model = cls.load_from_checkpoint(str(ckpt), map_location="cpu")
        model.eval().to(device)
        enc_p, enc_i = _branches(model)
        fam = "contrastive" if _is_contrastive(model) else "dual-encoder-fm"
        t_load = time.time() - t0

        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=nw, collate_fn=_collate, pin_memory=True)
        acc = {k: [] for k in ("hp", "hi", "lp", "li")}
        got = []
        t1 = time.time()
        for hsc, leg, r in loader:
            hsc, leg = hsc.to(device, non_blocking=True), leg.to(device, non_blocking=True)
            acc["hp"].append(enc_p(hsc).flatten(1).float().cpu().numpy())
            acc["hi"].append(enc_i(hsc).flatten(1).float().cpu().numpy())
            acc["lp"].append(enc_p(leg).flatten(1).float().cpu().numpy())
            acc["li"].append(enc_i(leg).flatten(1).float().cpu().numpy())
            got.append(r)
        cat = {k: np.concatenate(v, 0).astype(np.float32) for k, v in acc.items()}
        got = np.concatenate(got)

        if args.expect_width:
            for k, nm in (("hp", "physics"), ("hi", "instrument")):
                if cat[k].shape[1] != args.expect_width:
                    raise SystemExit(
                        f"{label}: {nm} width {cat[k].shape[1]} != expected {args.expect_width}. "
                        f"Refusing to write -- an avg-pooled arm in a conv1x1 comparison, or the "
                        f"wrong model class.")
        check_not_degenerate(f"{label}/hsc_physics", cat["hp"])
        check_not_degenerate(f"{label}/hsc_instrument", cat["hi"])

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
            f.attrs["model_class"] = f"{module}:{cls_name}"
            f.attrs["family"] = fam
        enc = time.time() - t1
        print(f"[arms] {n}/{len(todo)} {label:<38s} {fam:<16s} ckpt-load {t_load:5.1f}s  "
              f"encode {enc / 60:5.2f}m ({len(got) / enc:.0f} img/s)  -> {out.name}", flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
