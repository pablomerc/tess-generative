#!/usr/bin/env python
"""Build nested, parity-stratified training subsets for the data-scale ablation.

What this ablates: the number of CROSS-MATCHED galaxies (anchors), holding single-survey
coverage at full density. The same-instrument neighbour conditioning is drawn from the
disjoint pool rows (source_type 1/2), which are never anchors -- verified: the sky-kNN
graph is strictly bipartite anchors->pool, with zero anchor->anchor edges. So a random
anchor subset is the right design, and neighbours legitimately keep coming from the full
pool (that is what "abundant single-survey data" means).

Three properties this script guarantees:

  1. NESTED       -- scale_1000 subset of scale_3162 subset of scale_10000 subset of ...
                     One interleaved permutation, prefixes taken from it. Consecutive
                     rungs share galaxies, so the curve is not dominated by draw variance.
  2. PARITY-BALANCED -- the loader assigns anchor survey by raw position parity
                     (neighbors_ram48.py: anchor_is_hsc = idx % 2 == 0). Per-survey recon
                     MSE differs ~3.4x, so an unbalanced small draw would confound the
                     curve. Interleaving even/odd keeps every even-length prefix exactly
                     50/50.
  3. TRAIN-ONLY   -- positions are drawn from the seeded train split only, so the
                     validation set is identical (and untouched) in every arm.

Positions written to JSON are RAW anchor positions into the full dataset (pre-holdout),
so the files mean the same thing regardless of which exclusions are active. data_factory
translates them into kept-space.

Usage:
    python make_subsets.py                 # writes subsets/ + subsets/manifest.json
    python make_subsets.py --no-holdout    # reproduce the split without the eval holdout
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import random_split

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]

DEFAULT_DATA_DIR = "/orcd/pool/007/pablomer/efficient_neighs_48"
DEFAULT_HOLDOUT = (
    REPO
    / "galaxy_images/galaxy_model/downstream_evaluation/engaging/outputs/index/holdout_legacy_ids.txt"
)
# Half-decade ladder (10^3, 10^3.5, 10^4, 10^4.5). Even numbers so parity splits exactly.
DEFAULT_LADDER = [1000, 3162, 10000, 31622]


def _normalize_legacy_id(raw: str) -> str:
    """Mirror data_factory._normalize_legacy_id: strip the parquet's b'...' byte-repr."""
    s = str(raw)
    if len(s) >= 3 and s.startswith("b'") and s.endswith("'"):
        return s[2:-1]
    if len(s) >= 3 and s.startswith('b"') and s.endswith('"'):
        return s[2:-1]
    return s


def resolve_holdout_positions(holdout_txt: Path, data_dir: Path) -> tuple[list[int], int, int]:
    """Return (excluded anchor positions, n_anchors, n_holdout_ids) exactly as data_factory does."""
    cat = pd.read_parquet(data_dir / "catalog.parquet", columns=["object_id_legacy", "source_type"])
    is_anchor = cat["source_type"] == 0
    anchor_rows = cat.index[is_anchor].to_numpy()

    with open(holdout_txt) as f:
        holdout_ids = {ln.strip() for ln in f if ln.strip()}

    legacy_norm = cat["object_id_legacy"].astype(str).map(_normalize_legacy_id)
    matched_rows = cat.index[legacy_norm.isin(holdout_ids) & is_anchor].to_numpy()
    pos_lookup = {int(r): int(p) for p, r in enumerate(anchor_rows)}
    positions = sorted(pos_lookup[int(r)] for r in matched_rows)
    return positions, len(anchor_rows), len(holdout_ids)


def load_neighbor_tables(data_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (anchor_rows, neighbor_idx_hsc, neighbor_idx_legacy) over ALL catalog rows.

    Neighbour indices are absolute row indices into the full 468,197-row store, so they are
    keyed by catalog row (not by anchor position) -- hence anchor_rows for the translation.
    """
    cat = pd.read_parquet(
        data_dir / "catalog.parquet",
        columns=["source_type", "neighbor_idx_hsc", "neighbor_idx_legacy"],
    )
    anchor_rows = cat.index[cat["source_type"] == 0].to_numpy()
    nb_hsc = np.stack(cat["neighbor_idx_hsc"].to_numpy())
    nb_legacy = np.stack(cat["neighbor_idx_legacy"].to_numpy())
    return anchor_rows, nb_hsc, nb_legacy


def count_distinct_images(
    chosen_positions: np.ndarray,
    anchor_rows: np.ndarray,
    nb_hsc: np.ndarray,
    nb_legacy: np.ndarray,
    k: int,
) -> tuple[int, int]:
    """(distinct neighbour rows, total distinct images) for a rung.

    Mirrors the loader exactly: survey side by position parity (neighbors_ram48.py:71) and
    the first k valid slots (`nb[nb >= 0][:k]`, neighbors_ram48.py:90). Each anchor also
    contributes 2 images (its own row read from BOTH stores: target + cross-survey partner),
    while each neighbour row is read from one store only.

    This is the number that must appear alongside N in the paper table: we ablate
    cross-matches, but the model still touches single-survey pool images, and a reader who
    sees "N=1,000" and assumes 1,000 galaxies total would be misled.
    """
    rows = anchor_rows[chosen_positions]
    is_hsc = (chosen_positions % 2) == 0
    neighbour_rows: set[int] = set()
    for row, hsc in zip(rows, is_hsc):
        nb = nb_hsc[row] if hsc else nb_legacy[row]
        valid = nb[nb >= 0][:k]
        neighbour_rows.update(int(v) for v in valid)
    return len(neighbour_rows), 2 * len(chosen_positions) + len(neighbour_rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path(DEFAULT_DATA_DIR))
    ap.add_argument("--holdout-txt", type=Path, default=DEFAULT_HOLDOUT)
    ap.add_argument("--no-holdout", action="store_true", help="reproduce the split with no exclusions")
    ap.add_argument("--seed", type=int, default=42, help="must match trainer.seed")
    ap.add_argument("--val-ratio", type=float, default=0.05, help="must match data.val_ratio")
    ap.add_argument("--devices", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-steps", type=int, default=75000)
    ap.add_argument("--max-neighbors", type=int, default=5, help="must match data.max_neighbors")
    ap.add_argument("--ladder", type=int, nargs="+", default=DEFAULT_LADDER)
    ap.add_argument("--out-dir", type=Path, default=HERE / "subsets")
    args = ap.parse_args()

    # ---- reproduce data_factory's exclusion + split arithmetic -------------------------
    if args.no_holdout:
        excluded, total, n_ids = [], None, 0
        cat = pd.read_parquet(args.data_dir / "catalog.parquet", columns=["source_type"])
        total = int((cat["source_type"] == 0).sum())
    else:
        excluded, total, n_ids = resolve_holdout_positions(args.holdout_txt, args.data_dir)

    excluded_set = set(excluded)
    kept_positions = [i for i in range(total) if i not in excluded_set]
    kept_size = len(kept_positions)
    val_size = int(kept_size * args.val_ratio)
    train_size = kept_size - val_size

    g = torch.Generator().manual_seed(args.seed)
    train_split, val_split = random_split(range(kept_size), [train_size, val_size], generator=g)
    kept_arr = np.asarray(kept_positions)
    train_raw = kept_arr[np.asarray(train_split.indices)]
    val_raw = kept_arr[np.asarray(val_split.indices)]

    steps_per_epoch = math.ceil(train_size / args.devices) // args.batch_size

    print(f"anchors total            : {total:,}")
    if not args.no_holdout:
        print(f"holdout ids / excluded   : {n_ids:,} ids -> {len(excluded):,} anchors")
    print(f"kept / train / val       : {kept_size:,} / {train_size:,} / {val_size:,}")
    print(f"steps per epoch          : {steps_per_epoch}  ({args.num_steps / steps_per_epoch:.1f} epochs for {args.num_steps:,} steps)")

    # ---- interleaved, parity-stratified permutation of the train pool ------------------
    rng = np.random.default_rng(args.seed)
    evens = train_raw[train_raw % 2 == 0]
    odds = train_raw[train_raw % 2 == 1]
    rng.shuffle(evens)
    rng.shuffle(odds)
    pair_n = min(len(evens), len(odds))
    interleaved = np.empty(2 * pair_n, dtype=np.int64)
    interleaved[0::2] = evens[:pair_n]
    interleaved[1::2] = odds[:pair_n]
    # leftovers from the longer parity class go at the end (only reached by the full rung)
    leftover = np.concatenate([evens[pair_n:], odds[pair_n:]])
    order = np.concatenate([interleaved, leftover])
    assert len(order) == len(train_raw) and len(set(order.tolist())) == len(train_raw)
    print(f"train parity pool        : {len(evens):,} HSC-role / {len(odds):,} Legacy-role")

    # ---- write one JSON per rung, plus a manifest --------------------------------------
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print("  loading neighbour tables for distinct-image accounting ...")
    anchor_rows, nb_hsc, nb_legacy = load_neighbor_tables(args.data_dir)
    k = args.max_neighbors

    rungs = []
    for n in sorted(args.ladder):
        if n > len(order):
            raise SystemExit(f"ladder rung {n} exceeds train pool {len(order)}")
        chosen = np.sort(order[:n])
        n_even = int((chosen % 2 == 0).sum())
        n_nb, n_img = count_distinct_images(chosen, anchor_rows, nb_hsc, nb_legacy, k)
        out = args.out_dir / f"scale_{n}.json"
        with open(out, "w") as f:
            json.dump(
                {
                    "n_anchors": int(n),
                    "hsc_role": n_even,
                    "legacy_role": int(n) - n_even,
                    "frac_of_train": round(n / train_size, 6),
                    "distinct_neighbour_rows": n_nb,
                    "distinct_images_total": n_img,
                    "seed": args.seed,
                    "holdout_applied": not args.no_holdout,
                    "tile_to": train_size,
                    "positions": chosen.tolist(),
                },
                f,
            )
        rungs.append(
            {
                "n_anchors": int(n),
                "json": out.name,
                "hsc_role": n_even,
                "legacy_role": int(n) - n_even,
                "frac_of_train": round(n / train_size, 6),
                "repeats_per_epoch": round(train_size / n, 2),
                "distinct_neighbour_rows": n_nb,
                "distinct_images_total": n_img,
            }
        )
        print(
            f"  wrote {out.name:<20} n={n:>6,}  {n_even:>6,} HSC / {int(n) - n_even:>6,} Legacy  "
            f"{100 * n / train_size:5.2f}% of train  ~{train_size / n:6.1f} rep/epoch  "
            f"distinct images {n_img:>7,}"
        )

    # nestedness check
    for a, b in zip(sorted(args.ladder), sorted(args.ladder)[1:]):
        sa = set(json.load(open(args.out_dir / f"scale_{a}.json"))["positions"])
        sb = set(json.load(open(args.out_dir / f"scale_{b}.json"))["positions"])
        assert sa < sb, f"scale_{a} is not a strict subset of scale_{b}"
    print("  nestedness verified: every rung is a strict subset of the next")

    manifest = {
        "purpose": "data-scale ablation: number of cross-matched anchors, single-survey coverage held full",
        "data_dir": str(args.data_dir),
        "holdout_applied": not args.no_holdout,
        "holdout_txt": None if args.no_holdout else str(args.holdout_txt),
        "n_anchors_total": total,
        "n_excluded": len(excluded),
        "kept_size": kept_size,
        "train_size": train_size,
        "val_size": val_size,
        "seed": args.seed,
        "val_ratio": args.val_ratio,
        "devices": args.devices,
        "batch_size": args.batch_size,
        "effective_batch": args.devices * args.batch_size,
        "num_steps": args.num_steps,
        "steps_per_epoch": steps_per_epoch,
        "tile_to": train_size,
        "val_positions_raw_head": val_raw[:10].tolist(),
        "rungs": rungs,
    }
    with open(args.out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"  wrote manifest.json (tile_to={train_size:,}, steps/epoch={steps_per_epoch})")


if __name__ == "__main__":
    main()
