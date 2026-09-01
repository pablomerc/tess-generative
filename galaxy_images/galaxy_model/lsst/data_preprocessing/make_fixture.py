#!/usr/bin/env python3
"""Create fixtures/preprocess_reference.npz for the preprocessing self-test.

Run ONCE on the AMD cluster (hardcoded paths), then commit the ~1 MB fixture.
Picks 3 deterministic objects from data/lsst_efficient/ (2 COSMOS + 1
DEEP2-3), computes expected griz outputs with preprocessing.preprocess_lsst,
and cross-validates them against the independent implementation in
cross_matching/scripts/07_pixel_parity_gallery.py (the one validated by the
pixel-parity gallery). Only if both pipelines agree is the fixture written —
so the committed reference inherits the gallery validation, and the self-test
(`python -m galaxy_images.galaxy_model.lsst.preprocessing`) runs on any
cluster with no cross_matching dependency.
"""

from __future__ import annotations

import importlib.util
import json
import os
from datetime import date

import numpy as np
import torch

from galaxy_images.galaxy_model.lsst import preprocessing as pp

STORE = "/work1/jeroenaudenaert/pablomer/data/lsst_efficient"
SCRIPT07 = ("/work1/jeroenaudenaert/pablomer/cross_matching/scripts/"
            "07_pixel_parity_gallery.py")
# fixture lives next to preprocessing.py (one level up), which consumes it
FIXTURE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "fixtures", "preprocess_reference.npz")


def load_script07():
    spec = importlib.util.spec_from_file_location("script07", SCRIPT07)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    import pandas as pd

    with open(os.path.join(STORE, "meta.json")) as f:
        meta = json.load(f)
    M = meta["M"]
    mm = np.memmap(os.path.join(STORE, "lsst_image.bin"), dtype="float16",
                   mode="r", shape=tuple(meta["shape"]))
    tbl = pd.read_parquet(os.path.join(STORE, "lsst_meta.parquet"),
                          columns=["object_id_hsc", "field"])

    # deterministic picks: first COSMOS, middle COSMOS, first DEEP2-3
    cosmos = tbl.index[tbl.field == "COSMOS"]
    deep = tbl.index[tbl.field == "DEEP2-3"]
    rows = [int(cosmos[0]), int(cosmos[len(cosmos) // 2]), int(deep[0])]
    ids = tbl.loc[rows, "object_id_hsc"].tolist()
    print(f"fixture rows {rows} (ids {ids})")

    cubes16 = np.stack([np.asarray(mm[r][:4]) for r in rows])  # griz, f16
    assert not np.isnan(cubes16).any(), "griz must be complete"

    exp_un = pp.preprocess_lsst(cubes16, normalize=False)
    exp_no = pp.preprocess_lsst(cubes16, normalize=True)

    # cross-validate against the independent cross_matching implementation
    s07 = load_script07()
    mean, std = pp.LSST_NORM["griz"]
    for k in range(len(rows)):
        njy = cubes16[k].astype(np.float32) * pp.NJY_PER_NMGY
        ref_un = s07.preprocess_lsst(njy, normalize=False)
        ref_no = s07.preprocess_lsst(njy, norm=(mean, std))
        assert torch.allclose(exp_un[k], ref_un, atol=1e-4), \
            f"unnormalized mismatch vs script 07 (row {rows[k]})"
        assert torch.allclose(exp_no[k], ref_no, atol=1e-3), \
            f"normalized mismatch vs script 07 (row {rows[k]})"
    print("cross-validation vs cross_matching script 07: OK")

    os.makedirs(os.path.dirname(FIXTURE), exist_ok=True)
    np.savez_compressed(
        FIXTURE,
        input_griz_nmgy=cubes16,
        expected_unnormalized=exp_un.numpy(),
        expected_normalized=exp_no.numpy(),
        object_ids=np.asarray(ids, dtype=np.int64),
        store_rows=np.asarray(rows, dtype=np.int64),
        created=str(date.today()),
    )
    print(f"wrote {FIXTURE} ({os.path.getsize(FIXTURE) / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
