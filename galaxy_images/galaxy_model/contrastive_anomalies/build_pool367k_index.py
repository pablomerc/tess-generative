"""Cache the 367k paper pool as row indices into the 474k hsc_downstream corpus.

The paper's anomaly figure pool is ``source_type in {0, 1}`` of ``neighbours_v2.h5``
— 366,706 galaxies that carry a valid HSC stamp. Those rows are a *strict subset*
of the 474,954-row hsc_downstream store: all 366,706 object_ids are present, and
every one has a <1" positional match (median 0.008"). So no re-encoding is needed
to work on the paper pool — restrict the existing 474k latents to these rows.

Writes pool367k_rows.npy: sorted int64 hsc_downstream row indices.
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pyarrow.parquet as pq

NEIGHBOURS = Path("/orcd/pool/007/pablomer/neighbours_v2.h5")
HSCDOWN = Path("/home/pablomer/orcd/scratch/hsc_downstream")
OUT = Path("/orcd/pool/007/pablomer/outputs/contrastive_anomalies/pool367k_rows.npy")


def main():
    with h5py.File(NEIGHBOURS, "r") as f:
        st = f["source_type"][:]
        keep = (st == 0) | (st == 1)
        oid = f["object_id_hsc"][:][keep]
    n_pool = int(keep.sum())
    print(f"[pool] neighbours_v2 source_type in (0,1): {n_pool:,}")

    cat = pq.read_table(HSCDOWN / "catalog.parquet", columns=["object_id"])
    d_oid = cat["object_id"].to_numpy().astype(np.int64)

    want = np.array([int(x) for x in oid], dtype=np.int64)
    # map object_id -> hsc_downstream row
    order = np.argsort(d_oid, kind="stable")
    pos = np.searchsorted(d_oid[order], want)
    ok = (pos < len(order)) & (d_oid[order][np.clip(pos, 0, len(order) - 1)] == want)
    rows = order[pos[ok]]
    print(f"[pool] matched {ok.sum():,}/{n_pool:,} by object_id "
          f"(unmatched {int((~ok).sum()):,})")

    rows = np.unique(rows)
    print(f"[pool] distinct hsc_downstream rows: {len(rows):,}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.save(OUT, rows)
    print(f"[pool] wrote {OUT}")


if __name__ == "__main__":
    main()
