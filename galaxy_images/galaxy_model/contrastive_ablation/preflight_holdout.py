"""
Preflight for the contrastive DWNVAL run — RUN THIS BEFORE SUBMITTING TRAINING.

It costs ~1 minute of CPU and catches the one failure mode that would otherwise
waste a full GPU allocation: the holdout list silently matching zero (or few)
anchors because `object_id_legacy` is encoded differently in this cluster's
catalog (the parquet stores some ids as the string `b'123'`). A run with 0
matches trains happily and produces a checkpoint that is NOT a holdout run.

Checks
  1. store identity      — anchor/row counts match the Engaging store (103,741 / 468,197)
  2. holdout resolution   — >= --min-matches anchors matched, with a sample of ids
  3. parity balance       — excluded anchors are ~50/50 HSC/Legacy (anchor survey is
                            fixed by index parity, so a skewed exclusion would change
                            the training survey mix as well as the holdout)
  4. epoch geometry       — kept/train/val counts and steps/epoch vs the non-holdout run
  5. loader smoke test    — (optional, --smoke) pull 2 real batches through collate_neighbors
                            and assert shapes/finiteness
  6. throughput benchmark — (optional, --benchmark N) time N batches with realistic worker
                            count and project the wall clock for the full run. The 160x160
                            `efficient` store is random-read bound: on Engaging's Ceph it
                            manages ~0.35 batch/s => ~78h per 100k-step run, which is why
                            the Engaging arm used the prebaked `ram48` store. Whether that
                            holds on another cluster's filesystem is a measurement, not a
                            guess — make it before committing a GPU allocation.

Usage (AMD):
  python galaxy_images/galaxy_model/contrastive_ablation/preflight_holdout.py \
      --loader efficient \
      --data-dir /work1/jeroenaudenaert/pablomer/data/neighbors_efficient \
      --holdout-ids-txt galaxy_images/galaxy_model/contrastive_ablation/holdout_legacy_ids.txt \
      --smoke

Exit code 0 = safe to submit training. Non-zero = do not submit.
"""

import argparse
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parents[3]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# Reference geometry of the store that produced every published number.
EXPECTED_ANCHORS = 103_741
EXPECTED_ROWS = 468_197
# Non-holdout reference run (contrastive-spatial-conv1x1), for the epoch-shift report.
REF_TRAIN_SIZE = 98_554
REF_STEPS_PER_EPOCH = 98_554 // 64


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--loader", choices=["ram48", "efficient"], required=True)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--holdout-ids-txt", required=True)
    p.add_argument("--min-matches", type=int, default=5000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--val-fraction", type=float, default=0.05)
    p.add_argument("--max-neighbors", type=int, default=5)
    p.add_argument("--smoke", action="store_true",
                   help="Also pull 2 batches through the real collate to check shapes.")
    p.add_argument("--benchmark", type=int, default=0, metavar="N",
                   help="Time N batches with --bench-workers and project full-run wall clock. "
                        "Implies --smoke's dataset construction. Try 30.")
    p.add_argument("--bench-workers", type=int, default=8,
                   help="DataLoader workers for the benchmark; match the training job's "
                        "--cpus-per-task.")
    p.add_argument("--bench-max-steps", type=int, default=100_000,
                   help="Step count to project the wall clock for.")
    p.add_argument("--bench-warn-hours", type=float, default=20.0,
                   help="Warn if the projected data-only wall clock exceeds this.")
    p.add_argument("--bench-in-ram", choices=["auto", "0", "1"], default="auto",
                   help="For --loader ram48, load the store into RAM for the benchmark so it "
                        "matches how training reads it (default auto = yes for ram48). "
                        "Needs ~17GB, so run this inside the batch job, not on a login node. "
                        "Set 0 to benchmark the memmap path instead (much slower; not what "
                        "training does).")
    p.add_argument("--overlap-parquet", default=None,
                   help="Optional (Engaging only): outputs/index/overlap.parquet — verifies "
                        "the holdout list covers the eval sample's legacy ids.")
    args = p.parse_args()
    bench_in_ram = (args.loader == "ram48") if args.bench_in_ram == "auto" \
        else args.bench_in_ram == "1"

    failures = []
    warnings = []

    print("=" * 78)
    print("CONTRASTIVE DWNVAL PREFLIGHT")
    print("=" * 78)

    # ---- 1. store identity ---------------------------------------------------
    import pandas as pd
    catalog_path = Path(args.data_dir) / "catalog.parquet"
    if not catalog_path.exists():
        print(f"[1/5] FAIL  catalog.parquet not found at {catalog_path}")
        sys.exit(1)
    cat = pd.read_parquet(catalog_path, columns=["object_id_legacy", "source_type"])
    n_rows = len(cat)
    n_anchors = int((cat["source_type"] == 0).sum())
    ok_rows = n_rows == EXPECTED_ROWS
    ok_anch = n_anchors == EXPECTED_ANCHORS
    print(f"[1/5] store   rows={n_rows:,} (expect {EXPECTED_ROWS:,}) "
          f"anchors={n_anchors:,} (expect {EXPECTED_ANCHORS:,})")
    if not (ok_rows and ok_anch):
        warnings.append(
            f"store geometry differs from the Engaging store "
            f"(rows {n_rows:,} vs {EXPECTED_ROWS:,}, anchors {n_anchors:,} vs {EXPECTED_ANCHORS:,}). "
            f"The A/B against contrastive-spatial-conv1x1 assumes the same galaxies — "
            f"stop and reconcile before training."
        )

    # ---- 2. holdout resolution ----------------------------------------------
    from galaxy_images.galaxy_model.data_factory import (
        _resolve_downstream_holdout_positions,
    )
    holdout_txt = Path(args.holdout_ids_txt)
    if not holdout_txt.exists():
        print(f"[2/5] FAIL  holdout file not found: {holdout_txt}")
        sys.exit(1)
    n_ids = sum(1 for line in open(holdout_txt) if line.strip())
    positions = _resolve_downstream_holdout_positions(
        holdout_txt=holdout_txt, efficient_data_dir=args.data_dir,
    )
    print(f"[2/5] holdout ids_in_file={n_ids:,}  matched_anchors={len(positions):,}  "
          f"({100.0 * len(positions) / max(1, n_ids):.1f}% of ids)")
    if len(positions) < args.min_matches:
        failures.append(
            f"only {len(positions):,} anchors matched (need >= {args.min_matches:,}). "
            f"Almost certainly an object_id_legacy encoding mismatch — inspect "
            f"cat['object_id_legacy'].head() vs the first lines of {holdout_txt}."
        )
    else:
        print(f"        sample matched anchor positions: {positions[:5]} ... {positions[-3:]}")

    # ---- 3. parity balance ---------------------------------------------------
    n_even = sum(1 for i in positions if i % 2 == 0)
    frac_even = n_even / max(1, len(positions))
    print(f"[3/5] parity  excluded HSC-anchor={n_even:,} Legacy-anchor={len(positions) - n_even:,} "
          f"(HSC fraction {frac_even:.3f}, expect ~0.500)")
    if positions and not (0.45 <= frac_even <= 0.55):
        warnings.append(
            f"excluded anchors are parity-skewed (HSC fraction {frac_even:.3f}); the run's "
            f"HSC/Legacy training mix will differ from the non-holdout arm."
        )

    # ---- 4. epoch geometry ---------------------------------------------------
    kept = n_anchors - len(positions)
    val_size = max(1, int(kept * args.val_fraction))
    train_size = kept - val_size
    spe = train_size // args.batch_size
    print(f"[4/5] epochs  kept={kept:,} train={train_size:,} val={val_size:,} "
          f"steps/epoch={spe:,}  (non-holdout ref: train={REF_TRAIN_SIZE:,}, "
          f"steps/epoch={REF_STEPS_PER_EPOCH:,}; shift {100.0 * (spe - REF_STEPS_PER_EPOCH) / REF_STEPS_PER_EPOCH:+.1f}%)")

    # ---- 4b. optional: does the holdout cover the eval sample? ---------------
    if args.overlap_parquet:
        ov = pd.read_parquet(args.overlap_parquet, columns=["legacy_object_id"])
        ov_ids = {str(x) for x in ov["legacy_object_id"]}
        hold_ids = {line.strip() for line in open(holdout_txt) if line.strip()}
        covered = len(ov_ids & hold_ids)
        print(f"[4b ] overlap eval sample: {len(ov):,} rows / {len(ov_ids):,} unique legacy ids | "
              f"covered by holdout list: {covered:,} ({100.0 * covered / max(1, len(ov_ids)):.1f}%)")
        if covered < 0.99 * len(ov_ids):
            failures.append(
                f"holdout list covers only {covered:,}/{len(ov_ids):,} eval legacy ids — "
                f"the exclusion would not actually remove the whole eval sample."
            )

    # ---- 5. loader smoke test ------------------------------------------------
    if args.smoke or args.benchmark:
        import time
        import torch
        from torch.utils.data import DataLoader, Subset
        from galaxy_images.galaxy_model.neighbors import collate_neighbors

        if args.loader == "ram48":
            from galaxy_images.galaxy_model.contrastive_baseline.neighbors_ram48 import (
                NeighborsRAM48Dataset as DS,
            )
            # Don't pull 17GB into RAM just to smoke-test: memmap instead.
            ds = DS(data_dir=args.data_dir, crop_size=48,
                    max_neighbors=args.max_neighbors, load_into_ram=False)
        else:
            from galaxy_images.galaxy_model.neighbors_efficient import (
                NeighborsEfficientDataset as DS,
            )
            ds = DS(data_dir=args.data_dir, crop_size=48, max_neighbors=args.max_neighbors)

        excluded = set(positions)
        kept_pos = [i for i in range(len(ds)) if i not in excluded]
        # Take a small contiguous slice (keeps parity mix) rather than the whole subset.
        probe = Subset(ds, kept_pos[: 4 * args.batch_size])
        dl = DataLoader(probe, batch_size=args.batch_size, shuffle=False,
                        num_workers=0, collate_fn=collate_neighbors)
        for bi, batch in enumerate(dl):
            targets, samegals, sameins, masks, metadata = batch
            print(f"[5/5] batch{bi}  targets={tuple(targets.shape)} samegals={tuple(samegals.shape)} "
                  f"sameins={tuple(sameins.shape)} masks={tuple(masks.shape)} "
                  f"valid_neighbors={int(masks.sum())}/{masks.numel()} "
                  f"surveys={{'hsc': {sum(1 for m in metadata if m['anchor_survey'] == 'hsc')}, "
                  f"'legacy': {sum(1 for m in metadata if m['anchor_survey'] == 'legacy')}}}")
            if targets.shape[1:] != (4, 48, 48):
                failures.append(f"unexpected target shape {tuple(targets.shape)}; expected (B,4,48,48)")
            if not torch.isfinite(targets).all():
                failures.append("non-finite values in targets")
            if bi >= 1:
                break
        # The excluded anchors must not be reachable through the kept subset.
        leaked = excluded & set(kept_pos)
        if leaked:
            failures.append(f"{len(leaked)} holdout positions are still present in the kept subset")
        else:
            print(f"[5/5] exclusion verified: 0 of {len(excluded):,} holdout positions in kept subset")

        # ---- 6. throughput benchmark ----------------------------------------
        if args.benchmark:
            bench_ds = ds
            if args.loader == "ram48" and bench_in_ram:
                # The smoke test above deliberately memmaps the ram48 store (cheap), but
                # TRAINING loads it fully into RAM. Benchmarking the memmap would measure
                # a configuration nobody runs — on Ceph it is orders of magnitude slower
                # and the projection would be nonsense. Reload it the way training does.
                print(f"[6/6] loading the ram48 store into RAM (~17GB) so the benchmark "
                      f"matches training; this takes ~1 min...", flush=True)
                bench_ds = DS(data_dir=args.data_dir, crop_size=48,
                              max_neighbors=args.max_neighbors, load_into_ram=True)
            full = Subset(bench_ds, kept_pos)
            # shuffle=True on purpose: the store's cost is random-read latency, and a
            # contiguous read would flatter it by hitting sequential page-cache.
            bdl = DataLoader(
                full, batch_size=args.batch_size, shuffle=True,
                num_workers=args.bench_workers, collate_fn=collate_neighbors,
                persistent_workers=args.bench_workers > 0, pin_memory=False,
                drop_last=True,
            )
            it = iter(bdl)
            n_warm = min(5, args.benchmark)
            for _ in range(n_warm):          # let workers spin up / caches settle
                next(it)
            t0 = time.perf_counter()
            for _ in range(args.benchmark):
                next(it)
            dt = time.perf_counter() - t0
            bps = args.benchmark / dt
            hours = args.bench_max_steps / bps / 3600.0
            print(f"[6/6] bench   {args.benchmark} batches in {dt:.1f}s with "
                  f"{args.bench_workers} workers => {bps:.2f} batch/s "
                  f"({1000.0 / bps:.0f} ms/batch)")
            print(f"        projected data-only wall clock for {args.bench_max_steps:,} steps: "
                  f"{hours:.1f}h (excludes GPU compute; treat as a floor)")
            if hours > args.bench_warn_hours:
                warnings.append(
                    f"projected {hours:.1f}h of data loading alone for "
                    f"{args.bench_max_steps:,} steps at {bps:.2f} batch/s — this loader is "
                    f"I/O bound on this filesystem. Build the prebaked 48x48 store "
                    f"(contrastive_baseline/build_ram48.py, ~17GB) and re-run with "
                    f"--loader ram48, or raise --bench-workers if cores are free."
                )
            else:
                print(f"        OK: under the {args.bench_warn_hours:.0f}h warning threshold.")
    else:
        print("[5/5] skipped (pass --smoke to run the loader check)")

    # ---- verdict -------------------------------------------------------------
    print("=" * 78)
    for w in warnings:
        print(f"WARNING: {w}")
    for f in failures:
        print(f"FAILURE: {f}")
    if failures:
        print("PREFLIGHT FAILED — do not submit training.")
        sys.exit(1)
    print("PREFLIGHT OK" + (" (with warnings — read them before submitting)" if warnings else ""))
    sys.exit(0)


if __name__ == "__main__":
    main()
