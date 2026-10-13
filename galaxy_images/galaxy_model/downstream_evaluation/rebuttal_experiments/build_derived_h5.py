"""
Build a derived-feature copy of an engaging-pipeline embeddings H5.

Purpose: probe instrument properties from **concat(physics, instrument)** without
re-encoding anything. The cached H5s written by prepare_combined.py already hold
the four per-survey latents at (5469, 64), so every feature set we need is a
column-concatenation away.

The trick that keeps this zero-risk: we do NOT touch predict_combined.py. We write
a new H5 whose `combined_e2` dataset holds the derived feature vector, and run the
unmodified probe driver on it. `combined_e1` is copied through **verbatim** so its
probe scores must reproduce the published fold CSVs bit-for-bit — that is the
end-to-end validation gate (see aggregate_rebuttal_table.py --check-identity).

Recipes (all on the `overlap` subset, n=5469):

    e1e2_256    [hsc_e1 | legacy_e1 | hsc_e2 | legacy_e2]   256-D
                = 64 per image per latent x 2 surveys x 2 latents.
                Each 64 = 4 spatial tokens x 16 channels (the 2x2 grid of a
                48x48 input, flattened). The published "Ours" column is the
                right-hand half only: [hsc_e2 | legacy_e2] = 128-D.
    e1e2m_hsc   [hsc_e1 | hsc_e2]                           128-D  (HSC targets)
    e1e2m_leg   [legacy_e1 | legacy_e2]                     128-D  (Legacy targets)
    e2m_hsc     hsc_e2                                       64-D  (HSC targets)
    e2m_leg     legacy_e2                                    64-D  (Legacy targets)

The survey-matched recipes come in _hsc/_leg pairs because `combined_e2` is a single
dataset that predict_combined.py feeds to *both* instrument families; the mismatched
family's rows are cross-survey and get dropped at aggregation.

CLI:

    python build_derived_h5.py \
        --in  ../engaging/outputs/embeddings/base.h5 \
        --out outputs/embeddings/base-e1e2-256.h5 \
        --recipe e1e2_256 [--balance-blocks]

Modelled on ../engaging/build_meanpool_h5.py (same attr-copy conventions).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

SUBSET = "overlap"

# Recipe -> ordered source latent keys to concatenate along axis 1.
RECIPES: dict[str, tuple[str, ...]] = {
    "e1e2_256": ("hsc_e1", "legacy_e1", "hsc_e2", "legacy_e2"),
    "e1e2m_hsc": ("hsc_e1", "hsc_e2"),
    "e1e2m_leg": ("legacy_e1", "legacy_e2"),
    "e2m_hsc": ("hsc_e2",),
    "e2m_leg": ("legacy_e2",),
}

# Expected width, asserted so a silently-wrong source H5 fails loudly.
EXPECTED_DIM = {"e1e2_256": 256, "e1e2m_hsc": 128, "e1e2m_leg": 128,
                "e2m_hsc": 64, "e2m_leg": 64}

# Which instrument family the recipe is *valid* for. None = both. Recorded as an
# attr for provenance; the aggregator applies the same map when filtering rows.
KEEP_FAMILIES = {
    "e1e2_256": ["instrument_hsc", "instrument_legacy"],
    "e1e2m_hsc": ["instrument_hsc"],
    "e1e2m_leg": ["instrument_legacy"],
    "e2m_hsc": ["instrument_hsc"],
    "e2m_leg": ["instrument_legacy"],
}

BLOCK = 64  # width of one (survey, latent) block


def _blocks(arr: np.ndarray) -> list[np.ndarray]:
    if arr.shape[1] % BLOCK != 0:
        raise ValueError(f"width {arr.shape[1]} is not a multiple of {BLOCK}")
    return [arr[:, i:i + BLOCK] for i in range(0, arr.shape[1], BLOCK)]


def _balance_blocks(arr: np.ndarray) -> np.ndarray:
    """Divide each 64-D block by its own global RMS.

    The physics blocks carry ~1.9x the RMS of the instrument blocks (base.h5:
    hsc_e1 1.841 / legacy_e1 1.702 vs hsc_e2 0.922 / legacy_e2 0.967), so a raw
    concatenation hands the physics half more of the first layer's initial signal.
    One scalar per block is an unsupervised, label-independent rescale, so unlike a
    per-column z-score fitted on all rows it leaks essentially nothing about the
    held-out fold. Only used as the robustness re-run if the raw concat scores
    *below* the 128-D instrument-only baseline.
    """
    out = []
    for b in _blocks(arr):
        rms = float(np.sqrt((b.astype(np.float64) ** 2).mean()))
        out.append(b / rms if rms > 1e-12 else b)
    return np.concatenate(out, axis=1).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="src", required=True, type=Path)
    ap.add_argument("--out", dest="dst", required=True, type=Path)
    ap.add_argument("--recipe", required=True, choices=sorted(RECIPES))
    ap.add_argument("--balance-blocks", action="store_true",
                    help="Rescale each 64-D block to unit RMS before concatenating.")
    args = ap.parse_args()

    keys = RECIPES[args.recipe]
    args.dst.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(args.src, "r") as fin, h5py.File(args.dst, "w") as fout:
        if SUBSET not in fin:
            raise SystemExit(f"{args.src} has no /{SUBSET} subset")

        # ---- derived features -------------------------------------------------
        parts = []
        for k in keys:
            ds = f"{SUBSET}/{k}"
            if ds not in fin:
                raise SystemExit(f"{args.src} is missing {ds}")
            a = np.asarray(fin[ds][:], dtype=np.float32)
            if a.ndim != 2 or a.shape[1] != BLOCK:
                raise SystemExit(f"{ds} has shape {a.shape}, expected (N, {BLOCK})")
            parts.append(a)

        n = parts[0].shape[0]
        if any(p.shape[0] != n for p in parts):
            raise SystemExit(f"row-count mismatch across {keys}")

        feat = np.concatenate(parts, axis=1)
        if args.balance_blocks:
            feat = _balance_blocks(feat)

        want = EXPECTED_DIM[args.recipe]
        if feat.shape[1] != want:
            raise SystemExit(f"recipe {args.recipe} built width {feat.shape[1]}, expected {want}")
        if not np.isfinite(feat).all():
            raise SystemExit(f"recipe {args.recipe} produced non-finite features")

        grp = fout.create_group(SUBSET)
        grp.create_dataset("combined_e2", data=feat, compression="gzip", compression_opts=4)

        # combined_e1 copied verbatim -> its probe scores must match the published
        # fold CSVs exactly. This is the identity gate for the whole derived path.
        src_e1 = f"{SUBSET}/combined_e1"
        if src_e1 not in fin:
            raise SystemExit(f"{args.src} is missing {src_e1} (needed for the identity gate)")
        grp.create_dataset("combined_e1", data=np.asarray(fin[src_e1][:]),
                           compression="gzip", compression_opts=4)

        # ---- labels, verbatim -------------------------------------------------
        lab_in = fin[f"{SUBSET}/labels"]
        lab_out = grp.create_group("labels")
        for col in lab_in.keys():
            lab_out.create_dataset(col, data=np.asarray(lab_in[col][:]),
                                   compression="gzip", compression_opts=4)

        # ---- attrs ------------------------------------------------------------
        for k, v in fin.attrs.items():
            fout.attrs[k] = v

        def _s(key: str, default: str = "?") -> str:
            v = fin.attrs.get(key, default)
            return v.decode() if isinstance(v, bytes) else str(v)

        suffix = args.recipe.replace("_", "-") + ("-bal" if args.balance_blocks else "")
        fout.attrs["architecture"] = f"{_s('architecture')}-{suffix}"
        fout.attrs["checkpoint_name"] = f"{_s('checkpoint_name', args.src.stem)}-{suffix}"
        fout.attrs["has_e2"] = True
        # This subset is the only one written, so keep the counts honest.
        fout.attrs["num_examples_per_subset"] = json.dumps({SUBSET: int(n)})
        fout.attrs["label_columns_per_subset"] = json.dumps({SUBSET: sorted(lab_in.keys())})

        # Provenance.
        fout.attrs["latent_recipe"] = args.recipe
        fout.attrs["latent_recipe_keys"] = json.dumps(list(keys))
        fout.attrs["latent_dim"] = int(feat.shape[1])
        fout.attrs["latent_blocks_balanced"] = bool(args.balance_blocks)
        fout.attrs["keep_families"] = json.dumps(KEEP_FAMILIES[args.recipe])
        fout.attrs["source_h5"] = str(args.src.resolve())
        fout.attrs["source_checkpoint_path"] = _s("checkpoint_path")
        fout.attrs["derived_by"] = "rebuttal_experiments/build_derived_h5.py"

        print(f"[build] {args.dst.name}")
        print(f"        recipe={args.recipe}  keys={' | '.join(keys)}")
        print(f"        combined_e2 {feat.shape}  rms={np.sqrt((feat.astype(np.float64)**2).mean()):.4f}"
              f"  balanced={args.balance_blocks}")
        print(f"        combined_e1 {fin[src_e1].shape} (verbatim)  labels={len(lab_in.keys())}")
        print(f"        keep_families={KEEP_FAMILIES[args.recipe]}")


if __name__ == "__main__":
    main()
