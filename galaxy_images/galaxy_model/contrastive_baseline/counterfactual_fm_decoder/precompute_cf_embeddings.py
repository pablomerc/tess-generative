"""
Stage 1 — precompute counterfactual [embedding, image] H5 for the CF FM decoder.

For each anchor we NEVER encode the target. Instead:

    emb_galaxy      (N, 64)        encoder_galaxy(samegal)         — physics from other survey
    emb_instrument  (N, K, 64)     encoder_instrument(sameins[k])  — instrument from neighbors
    masks           (N, K) bool    True = real neighbor, False = pad
    targets         (N, 4, 48, 48) the anchor image to reconstruct
    survey          (N,) int8      0 = hsc, 1 = legacy
    row_idx         (N,) int64     source catalog row of the anchor

Reads the anchors in dataset order (shuffle=False) so H5 row i corresponds to
dataset position i, which is what makes random_split(val_ratio=0.05, seed=42)
reproduce the data_factory / paper / fm-control split exactly.

DEFAULT LOADER IS ``ram48`` (the compact in-RAM 48x48 store). This matters for
three independent reasons:

  1. Speed. The 160x160 ``efficient_neighs`` mmap store needs 7 random Ceph row
     reads per anchor (anchor-hsc + anchor-legacy + K neighbors) and delivered
     ~1 anchor/s in job 18875467 => ~29 h for 104k anchors. ram48 is a
     sequential 16 GB slurp into RAM, after which the pass is GPU-bound.
  2. Encoder fidelity. train_contrastive_engaging.py defaults to
     ``efficient_neighs_48``, so the frozen dual-encoder was fit on the float16
     ram48 tensors. Feeding it ram48 keeps inference on its training inputs.
  3. Eval comparability. eval_cf_recon_mse.py scores the image-conditioned
     baselines against the ram48 loader's x_1 and the CF decoder against the
     targets stored here. Building from ram48 makes those the SAME tensors.

Both stores share catalog.parquet (build_ram48.py copies it), so anchor order,
anchor count and the idx%2 survey assignment are identical either way.

Writes to ``<out-h5>.tmp`` then renames on success. Optional milestone snapshots
(``--snapshot-at``) write finalized standalone H5s so a timeout still leaves a
usable corpus; with the ram48 loader the full pass is short enough that these
are off by default.

Usage:

    python -m galaxy_images.galaxy_model.contrastive_baseline.counterfactual_fm_decoder.precompute_cf_embeddings \
        --data-dir /orcd/pool/007/pablomer/efficient_neighs_48 --loader ram48 \
        --ckpt     /orcd/pool/007/pablomer/outputs/contrastive/best-epoch=599-step=240000.ckpt \
        --out-h5   /orcd/pool/007/pablomer/outputs/counterfactual_fm/cf_embeddings_all.h5
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import h5py
import numpy as np
import torch

try:
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "preferred_blas_library"):
        torch.backends.cuda.preferred_blas_library("hipblas")
except Exception:
    pass
from torch.utils.data import DataLoader, Subset

_project_root = Path(__file__).resolve().parents[4]  # .../tess-generative
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from galaxy_images.galaxy_model.contrastive_baseline.dual_encoder_contrastive import (
    DualEncoderContrastiveModule,
)

DEFAULT_DATA_DIR = "/orcd/pool/007/pablomer/efficient_neighs_48"
DEFAULT_CKPT = "/orcd/pool/007/pablomer/outputs/contrastive/best-epoch=599-step=240000.ckpt"
DEFAULT_OUT = "/orcd/pool/007/pablomer/outputs/counterfactual_fm/cf_embeddings_all.h5"


def _collate(batch, max_neighbors: int, crop_size: int):
    """Pad sameins to K; return stacked tensors + metadata lists."""
    B = len(batch)
    targets = torch.stack([b[0] for b in batch])
    samegals = torch.stack([b[1] for b in batch])
    sameins = torch.zeros(B, max_neighbors, 4, crop_size, crop_size, dtype=torch.float32)
    masks = torch.zeros(B, max_neighbors, dtype=torch.bool)
    surveys = []
    row_idxs = []
    for i, (target, samegal, si, meta) in enumerate(batch):
        k = min(si.shape[0], max_neighbors)
        if k > 0:
            sameins[i, :k] = si[:k]
            masks[i, :k] = True
        surveys.append(meta["anchor_survey"])
        row_idxs.append(int(meta["idx"]))
    return targets, samegals, sameins, masks, surveys, row_idxs


def _build_dataset(loader: str, data_dir: str, crop_size: int, K: int, load_into_ram: bool):
    """Build the anchor dataset. Both loaders share the (target, samegal, sameins, meta) contract."""
    if loader == "ram48":
        from galaxy_images.galaxy_model.contrastive_baseline.neighbors_ram48 import (
            NeighborsRAM48Dataset,
        )
        return NeighborsRAM48Dataset(
            data_dir, crop_size=crop_size, max_neighbors=K, load_into_ram=load_into_ram,
        )
    if loader == "efficient":
        from galaxy_images.galaxy_model.neighbors_efficient import NeighborsEfficientDataset
        return NeighborsEfficientDataset(data_dir, crop_size=crop_size, max_neighbors=K)
    raise ValueError(f"unknown loader {loader!r}; use ram48|efficient")


def _load_encoder(ckpt_path: str, device: torch.device) -> DualEncoderContrastiveModule:
    model = DualEncoderContrastiveModule.load_from_checkpoint(ckpt_path, map_location="cpu")
    model.eval()
    model.to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def _encoder_out_dim(model: DualEncoderContrastiveModule) -> int:
    """True encoder output width.

    ``hparams.embedding_dim`` only equals the output width for encoder_pool="avg"
    and for "conv1x1" with the default token_dim; reading out_dim makes a conv1x1
    checkpoint with an explicit encoder_token_dim fail loudly here instead of as a
    cryptic h5py shape error one batch later.
    """
    dims = []
    for name in ("encoder_galaxy", "encoder_instrument"):
        enc = getattr(model, name)
        d = getattr(enc, "out_dim", None)
        dims.append(int(d if d is not None else model.hparams.embedding_dim))
    if dims[0] != dims[1]:
        raise ValueError(
            f"encoder_galaxy.out_dim={dims[0]} != encoder_instrument.out_dim={dims[1]}; "
            f"the CF conditioning assumes a shared embedding width."
        )
    return dims[0]


def _parse_snapshots(s: str) -> List[int]:
    if not s.strip():
        return []
    out = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return sorted(set(out))


def _ping(webhook: str, content: str) -> None:
    if not webhook:
        return
    try:
        req = urllib.request.Request(
            webhook,
            data=json.dumps({"content": content}).encode(),
            headers={
                "Content-Type": "application/json",
                "User-Agent": "Mozilla/5.0 (compatible; cf-precompute)",
            },
        )
        urllib.request.urlopen(req, timeout=10)
    except Exception as exc:  # noqa: BLE001
        print(f"[precompute-cf] discord ping failed (ignored): {exc}", flush=True)


def _set_common_attrs(
    f: h5py.File,
    *,
    num_examples: int,
    emb_dim: int,
    K: int,
    data_dir: str,
    ckpt: str,
    crop_size: int,
    loader: str,
    n_anchors_total: int,
    target_dtype: str,
    snapshot_of: Optional[int] = None,
) -> None:
    """Stamp the finalized-file attrs.

    ``complete`` is the contract with CFEmbeddingDataset: it is only ever set on a
    file that has been fully written, so a reader can refuse a truncated corpus
    instead of silently training on pre-allocated zero rows.
    ``n_anchors_total`` lets the trainer tell a full corpus (paper-matched 5% val
    split) from a snapshot (split does NOT match) — see train_cf_fm.py.
    """
    f.attrs["num_examples"] = num_examples
    f.attrs["embedding_dim_each"] = emb_dim
    f.attrs["max_neighbors"] = K
    f.attrs["cond_dim_meanpool"] = 2 * emb_dim
    f.attrs["n_tokens"] = 1 + K
    f.attrs["source_data_dir"] = data_dir
    f.attrs["source_loader"] = loader
    f.attrs["contrastive_ckpt"] = ckpt
    f.attrs["crop_size"] = crop_size
    f.attrs["counterfactual"] = True
    f.attrs["n_anchors_total"] = n_anchors_total
    f.attrs["target_dtype"] = target_dtype
    f.attrs["complete"] = True
    if snapshot_of is not None:
        f.attrs["snapshot_n"] = snapshot_of
        f.attrs["is_snapshot"] = True


def _write_snapshot(
    src: h5py.File,
    n_keep: int,
    snap_path: Path,
    *,
    emb_dim: int,
    K: int,
    data_dir: str,
    ckpt: str,
    crop_size: int,
    chunk: int,
    loader: str,
    n_anchors_total: int,
    target_dtype: str,
) -> None:
    """Flush working file and write a finalized standalone H5 with the first n_keep rows."""
    src.flush()
    snap_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = snap_path.with_suffix(snap_path.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()
    print(f"[precompute-cf] writing snapshot n={n_keep:,} -> {snap_path}", flush=True)
    t0 = time.time()
    with h5py.File(tmp, "w") as out:
        for name in ("targets", "emb_galaxy", "emb_instrument", "masks", "survey", "row_idx"):
            src_ds = src[name]
            shape = (n_keep,) + src_ds.shape[1:]
            kwargs = dict(dtype=src_ds.dtype, compression="lzf")
            if name == "targets":
                kwargs["chunks"] = (min(chunk, n_keep),) + src_ds.shape[1:]
            elif name == "emb_instrument":
                kwargs["chunks"] = (min(chunk, n_keep),) + src_ds.shape[1:]
            dst = out.create_dataset(name, shape=shape, **kwargs)
            # Copy in slabs to avoid a giant RAM buffer.
            step = max(1, min(chunk * 8, n_keep))
            for i0 in range(0, n_keep, step):
                i1 = min(n_keep, i0 + step)
                dst[i0:i1] = src_ds[i0:i1]
        _set_common_attrs(
            out,
            num_examples=n_keep,
            emb_dim=emb_dim,
            K=K,
            data_dir=data_dir,
            ckpt=ckpt,
            crop_size=crop_size,
            loader=loader,
            n_anchors_total=n_anchors_total,
            target_dtype=target_dtype,
            snapshot_of=n_keep,
        )
    tmp.replace(snap_path)
    print(
        f"[precompute-cf] snapshot ready {snap_path} ({n_keep:,} rows) in "
        f"{time.time() - t0:.1f}s",
        flush=True,
    )


def main():
    ap = argparse.ArgumentParser(description="Precompute CF embeddings (samegal + neighbors).")
    ap.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    ap.add_argument("--loader", choices=["ram48", "efficient"], default="ram48",
                    help="ram48 = compact in-RAM 48x48 store: ~30x faster, matches the frozen "
                         "encoder's training inputs AND the eval loader's targets. "
                         "efficient = 160x160 Ceph mmap store (~1 anchor/s; provenance checks only).")
    ap.add_argument("--no-ram", action="store_true",
                    help="ram48 loader only: mmap instead of slurping into RAM (debug).")
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--out-h5", default=DEFAULT_OUT)
    ap.add_argument("--limit", type=int, default=0,
                    help="If >0, encode only the first N anchors. Default 0 = all anchors. "
                         "A limited H5 does NOT reproduce the paper val split.")
    ap.add_argument("--max-neighbors", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--crop-size", type=int, default=48)
    ap.add_argument("--target-dtype", choices=["float16", "float32"], default="float16",
                    help="On-disk dtype for targets. float16 is lossless w.r.t. the ram48 store "
                         "(already float16) and halves the file (~3.8 GB -> ~1.9 GB).")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument(
        "--snapshot-at",
        default="",
        help="Comma-separated cumulative counts at which to write finalized partial H5s "
             "(same dir as --out-h5, named cf_embeddings_<N>k.h5 / cf_embeddings_<N>.h5). "
             "Empty (default) = off; with --loader ram48 the full pass is short enough that "
             "milestone insurance costs more disk than it saves.",
    )
    ap.add_argument(
        "--discord-webhook",
        default="",
        help="Optional Discord webhook for snapshot / completion pings.",
    )
    args = ap.parse_args()

    device = torch.device(args.device)
    K = args.max_neighbors
    snapshots = [s for s in _parse_snapshots(args.snapshot_at) if s > 0]
    print(f"[precompute-cf] device={device} | K={K} | snapshots={snapshots}")
    print(f"[precompute-cf] loader={args.loader} | data_dir={args.data_dir}")
    print(f"[precompute-cf] ckpt={args.ckpt}")
    model = _load_encoder(args.ckpt, device)
    emb_dim = _encoder_out_dim(model)
    print(
        f"[precompute-cf] encoder out_dim={emb_dim} "
        f"(hparams.embedding_dim={int(model.hparams.embedding_dim)}, "
        f"pool={getattr(model.hparams, 'encoder_pool', '?')})"
    )

    ds = _build_dataset(
        args.loader, args.data_dir, args.crop_size, K, load_into_ram=not args.no_ram,
    )
    n_total = len(ds)
    n = n_total if args.limit <= 0 else min(args.limit, n_total)
    print(f"[precompute-cf] {n_total:,} anchors available; encoding {n:,}")
    if n != n_total:
        print(
            "[precompute-cf] WARNING: partial run — the resulting H5 will NOT reproduce the "
            "data_factory/paper 5% val split. Train on it only with --allow-partial-h5."
        )
    subset = Subset(ds, list(range(n)))

    loader = DataLoader(
        subset,
        batch_size=args.batch_size,
        # H5 row i must equal dataset position i, or the seed-42 split stops matching
        # data_factory and the eval val set leaks into CF training.
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=lambda b: _collate(b, K, args.crop_size),
        pin_memory=(device.type == "cuda"),
    )

    out_path = Path(args.out_h5)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    if tmp_path.exists():
        print(f"[precompute-cf] removing stale {tmp_path}", flush=True)
        tmp_path.unlink()
    if out_path.exists():
        # Avoid leaving a previous corrupt/partial final in place mid-run.
        bak = out_path.with_suffix(out_path.suffix + ".bak")
        print(f"[precompute-cf] moving existing {out_path} -> {bak}", flush=True)
        if bak.exists():
            bak.unlink()
        out_path.rename(bak)

    pending_snaps = [s for s in snapshots if s <= n]
    done_snaps: set[int] = set()

    def snap_path_for(count: int) -> Path:
        if count % 1000 == 0:
            return out_path.parent / f"cf_embeddings_{count // 1000}k.h5"
        return out_path.parent / f"cf_embeddings_{count}.h5"

    stat = {0: [0, 0.0, 0.0], 1: [0, 0.0, 0.0]}
    t0 = time.time()
    chunk = min(args.batch_size, n)
    n_real_neighbors = 0

    with h5py.File(tmp_path, "w") as f:
        d_tgt = f.create_dataset(
            "targets", shape=(n, 4, args.crop_size, args.crop_size),
            dtype=args.target_dtype, compression="lzf",
            chunks=(chunk, 4, args.crop_size, args.crop_size),
        )
        d_g = f.create_dataset("emb_galaxy", shape=(n, emb_dim), dtype="float32", compression="lzf")
        d_i = f.create_dataset(
            "emb_instrument", shape=(n, K, emb_dim), dtype="float32", compression="lzf",
            chunks=(chunk, K, emb_dim),
        )
        d_m = f.create_dataset("masks", shape=(n, K), dtype="bool", compression="lzf")
        d_s = f.create_dataset("survey", shape=(n,), dtype="int8")
        d_r = f.create_dataset("row_idx", shape=(n,), dtype="int64")

        ptr = 0
        with torch.no_grad():
            for bi, (targets, samegals, sameins, masks, surveys, row_idxs) in enumerate(loader):
                B = targets.shape[0]
                sg = samegals.to(device, dtype=torch.float32, non_blocking=True)
                si = sameins.to(device, dtype=torch.float32, non_blocking=True)
                m = masks.to(device)

                eg = model.encoder_galaxy(sg).float()  # (B, D)
                if eg.shape[1] != emb_dim:
                    raise RuntimeError(
                        f"encoder_galaxy returned width {eg.shape[1]}, expected {emb_dim}"
                    )

                si_flat = si.view(B * K, 4, args.crop_size, args.crop_size)
                ei_flat = model.encoder_instrument(si_flat).float()  # (B*K, D)
                ei = ei_flat.view(B, K, emb_dim)
                ei = ei * m.unsqueeze(-1).float()  # zero pad slots

                tnp = targets.numpy().astype(np.float32)
                surv = np.array([0 if s == "hsc" else 1 for s in surveys], dtype=np.int8)

                d_tgt[ptr:ptr + B] = tnp.astype(args.target_dtype)
                d_g[ptr:ptr + B] = eg.cpu().numpy()
                d_i[ptr:ptr + B] = ei.cpu().numpy()
                d_m[ptr:ptr + B] = masks.numpy()
                d_s[ptr:ptr + B] = surv
                d_r[ptr:ptr + B] = np.array(row_idxs, dtype=np.int64)
                ptr += B
                n_real_neighbors += int(masks.sum().item())

                for code in (0, 1):
                    sel = tnp[surv == code]
                    if sel.size:
                        stat[code][0] += sel.size
                        stat[code][1] += float(sel.sum())
                        stat[code][2] += float(np.square(sel).sum())

                # Emit any due snapshots (handles batches that jump past a boundary).
                for thresh in pending_snaps:
                    if thresh in done_snaps:
                        continue
                    if ptr >= thresh:
                        sp = snap_path_for(thresh)
                        _write_snapshot(
                            f, thresh, sp,
                            emb_dim=emb_dim, K=K,
                            data_dir=str(args.data_dir), ckpt=str(args.ckpt),
                            crop_size=args.crop_size, chunk=chunk,
                            loader=str(args.loader), n_anchors_total=n_total,
                            target_dtype=str(args.target_dtype),
                        )
                        done_snaps.add(thresh)
                        _ping(
                            args.discord_webhook,
                            f"cf-precompute 📸 snapshot `{sp.name}` ready "
                            f"({thresh:,} / {n:,}) · job continuing",
                        )

                if bi % 20 == 0:
                    elapsed = max(1e-9, time.time() - t0)
                    rate = ptr / elapsed
                    eta_min = (n - ptr) / max(1e-9, rate) / 60.0
                    print(
                        f"[precompute-cf] {ptr:,}/{n:,}  ({rate:.1f} anch/s)  "
                        f"eta={eta_min:.1f} min",
                        flush=True,
                    )

        if ptr != n:
            raise RuntimeError(
                f"wrote {ptr} rows but allocated {n}; refusing to stamp complete=True"
            )
        _set_common_attrs(
            f,
            num_examples=ptr,
            emb_dim=emb_dim,
            K=K,
            data_dir=str(args.data_dir),
            ckpt=str(args.ckpt),
            crop_size=args.crop_size,
            loader=str(args.loader),
            n_anchors_total=n_total,
            target_dtype=str(args.target_dtype),
        )
        f.attrs["n_real_neighbors"] = n_real_neighbors
        f.flush()

    tmp_path.replace(out_path)
    dt = time.time() - t0
    print(f"[precompute-cf] wrote {n:,} examples to {out_path} in {dt / 60:.1f} min")
    print(
        f"[precompute-cf] neighbor slots filled: {n_real_neighbors:,}/{n * K:,} "
        f"({100.0 * n_real_neighbors / max(1, n * K):.2f}%)"
    )
    if done_snaps:
        print(f"[precompute-cf] snapshots written: {sorted(done_snaps)}", flush=True)
    _ping(
        args.discord_webhook,
        f"cf-precompute ✅ full H5 ready `{out_path}` ({n:,}) · "
        f"snapshots={sorted(done_snaps)}",
    )
    for code, name in ((0, "hsc"), (1, "legacy")):
        c, s, ss = stat[code]
        if c:
            mean = s / c
            var = max(0.0, ss / c - mean * mean)
            print(
                f"[precompute-cf] target[{name}] mean={mean:+.4f} std={var ** 0.5:.4f} "
                f"(expected ~0 mean / ~1 std after per-survey normalization)"
            )


if __name__ == "__main__":
    main()
