"""Shared data access for the contrastive anomaly-detection experiment.

The corpus is the ``source_type == 0`` anchor set of the ram48 store
(``/orcd/pool/007/pablomer/efficient_neighs_48``) — 103,741 galaxies, each with a
matched HSC *and* Legacy Survey 48x48 4-band cutout. That is the same corpus the
contrastive dual-encoder was trained on and the same one the FM decoders
(``contrastive_baseline/{embedding,counterfactual}_fm_decoder``) precomputed
embeddings for, so the encoder is fed exactly its training-time inputs.

Two things this module exists to get right:

1. **Reading the store without hammering Ceph.** ``NeighborsRAM48Dataset`` slurps
   both 8.6 GB bins into RAM (17.2 GB) because *training* needs random access to
   all 468,197 rows. We only need the 103,741 anchors, so we do one strictly
   sequential pass in blocks and keep only anchor rows -> 3.8 GB resident and no
   random Ceph reads. Sequential is the only access pattern this filesystem is
   good at (see the perf note in ``contrastive_baseline/neighbors_ram48.py``).

2. **Lens ground truth.** ``/orcd/pool/007/pablomer/lens_indices.txt`` holds 74
   known strong lenses by ``hsc_object_id``; all 74 are anchors in this corpus
   (verified by both object-id join and the file's own ``absolute_index``), which
   is what makes a quantitative recall/AUC metric possible instead of eyeballing
   stamp grids.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np
import pandas as pd

DEFAULT_DATA_DIR = "/orcd/pool/007/pablomer/efficient_neighs_48"
DEFAULT_LENS_FILE = "/orcd/pool/007/pablomer/lens_indices.txt"

# Read the bins in blocks of this many source rows. 8192 x 4 x 48 x 48 x 2 B
# = 151 MB per block -- large enough that the read is sequential-throughput
# bound, small enough to stay off the memory ceiling.
_BLOCK_ROWS = 8192


def _decode_ids(series: pd.Series) -> np.ndarray:
    """Catalog object ids arrive as str (HSC) or bytes (Legacy); normalise to str."""
    return np.array(
        [v.decode() if isinstance(v, (bytes, bytearray)) else str(v) for v in series],
        dtype=object,
    )


@dataclass
class AnchorCatalog:
    """Per-anchor metadata, ordered by ascending source-catalog row."""

    row_idx: np.ndarray        # (N,) int64 — row in the 468,197-row store
    ra: np.ndarray             # (N,) float64 degrees
    dec: np.ndarray            # (N,) float64 degrees
    object_id_hsc: np.ndarray  # (N,) object (str)
    object_id_legacy: np.ndarray
    is_lens: np.ndarray        # (N,) bool
    n_rows_total: int          # rows in the underlying store

    def __len__(self) -> int:
        return len(self.row_idx)


def load_anchor_catalog(
    data_dir: str | Path = DEFAULT_DATA_DIR,
    lens_file: str | Path | None = DEFAULT_LENS_FILE,
    limit: int | None = None,
) -> AnchorCatalog:
    """Load anchor metadata + the strong-lens flag.

    ``limit`` truncates to the first N anchors in catalog-row order. Note that
    truncating will usually drop some of the 74 lenses, so the lens metrics get
    noisier — prefer running the full 103,741.
    """
    data_dir = Path(data_dir)
    cat = pd.read_parquet(
        data_dir / "catalog.parquet",
        columns=["source_type", "ra", "dec", "object_id_hsc", "object_id_legacy"],
    )
    n_rows_total = len(cat)
    anchor_mask = cat["source_type"].to_numpy() == 0
    row_idx = np.flatnonzero(anchor_mask).astype(np.int64)
    if limit is not None:
        row_idx = row_idx[:limit]

    sub = cat.iloc[row_idx]
    oid_hsc = _decode_ids(sub["object_id_hsc"])
    oid_legacy = _decode_ids(sub["object_id_legacy"])

    is_lens = np.zeros(len(row_idx), dtype=bool)
    if lens_file is not None and Path(lens_file).exists():
        lens = pd.read_csv(lens_file, sep="\t")
        wanted = {str(v) for v in lens["hsc_object_id"]}
        is_lens = np.array([o in wanted for o in oid_hsc], dtype=bool)
        # Join on hsc_object_id only. The file's `absolute_index` column indexes a
        # DIFFERENT store (it is neither a catalog row nor an anchor position here),
        # so using it would silently mislabel the ground truth.
        n_found = int(is_lens.sum())
        if limit is None and n_found != len(wanted):
            raise ValueError(
                f"lens ground truth: matched {n_found} of {len(wanted)} ids from {lens_file} "
                f"among {len(row_idx):,} anchors. All of them are expected to be present; "
                "refusing to run on a partial ground truth."
            )
        print(f"[anchor_data] lens ground truth: {n_found} / {len(wanted)} lenses "
              f"present among {len(row_idx):,} anchors", flush=True)

    return AnchorCatalog(
        row_idx=row_idx,
        ra=sub["ra"].to_numpy().astype(np.float64),
        dec=sub["dec"].to_numpy().astype(np.float64),
        object_id_hsc=oid_hsc,
        object_id_legacy=oid_legacy,
        is_lens=is_lens,
        n_rows_total=n_rows_total,
    )


def load_anchor_images(
    data_dir: str | Path,
    survey: str,
    row_idx: np.ndarray,
    n_rows_total: int,
    crop_size: int = 48,
    n_bands: int = 4,
    dtype: str = "float16",
    verbose: bool = True,
) -> np.ndarray:
    """Sequentially read one survey's bin and return only the anchor rows.

    Returns ``(len(row_idx), n_bands, crop_size, crop_size)`` float16 in the same
    order as ``row_idx``. ``row_idx`` must be sorted ascending (it is, coming out
    of ``load_anchor_catalog``).
    """
    data_dir = Path(data_dir)
    fname = {"hsc": "hsc48.bin", "legacy": "legacy48.bin"}[survey]
    path = data_dir / fname
    itemsize = np.dtype(dtype).itemsize
    row_elems = n_bands * crop_size * crop_size
    row_bytes = row_elems * itemsize

    expect = n_rows_total * row_bytes
    actual = path.stat().st_size
    if actual != expect:
        raise ValueError(f"{path}: expected {expect} bytes for {n_rows_total} rows, found {actual}")

    if not np.all(np.diff(row_idx) > 0):
        raise ValueError("row_idx must be strictly ascending for the sequential reader")

    out = np.empty((len(row_idx), n_bands, crop_size, crop_size), dtype=dtype)
    written = 0
    # row_idx is sorted, so a single scan with a moving cursor picks out each
    # block's anchors without ever searching backwards.
    for start in range(0, n_rows_total, _BLOCK_ROWS):
        stop = min(start + _BLOCK_ROWS, n_rows_total)
        lo = np.searchsorted(row_idx, start, side="left")
        hi = np.searchsorted(row_idx, stop, side="left")
        if lo == hi:
            continue
        block = np.fromfile(
            path, dtype=dtype, count=(stop - start) * row_elems, offset=start * row_bytes
        ).reshape(stop - start, n_bands, crop_size, crop_size)
        out[lo:hi] = block[row_idx[lo:hi] - start]
        written += hi - lo
        if verbose and (lo // 20000) != ((lo + (hi - lo)) // 20000):
            print(f"[anchor_data] {survey}: {written:,}/{len(row_idx):,} anchors read", flush=True)

    if written != len(row_idx):
        raise RuntimeError(f"read {written} anchors, expected {len(row_idx)}")
    if verbose:
        print(f"[anchor_data] {survey}: {out.nbytes / 1e9:.2f} GB resident", flush=True)
    return out


def store_meta(data_dir: str | Path = DEFAULT_DATA_DIR) -> dict:
    with open(Path(data_dir) / "meta.json") as f:
        return json.load(f)
