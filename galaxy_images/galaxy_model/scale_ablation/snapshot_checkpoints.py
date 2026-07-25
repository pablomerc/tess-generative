#!/usr/bin/env python
"""Archive rolling training checkpoints before ModelCheckpoint deletes them.

WHY THIS EXISTS
---------------
train.py's periodic ModelCheckpoint uses save_top_k=1 with monitor=None, so Lightning
DELETES the previous file every time it writes a new one (see
pytorch_lightning/callbacks/model_checkpoint.py::_save_none_monitor_checkpoint). Only ONE
`latest-step=step=<N>.ckpt` ever exists per run. That makes an R^2-vs-step curve impossible
after the fact -- the intermediate checkpoints are simply gone.

This poller copies checkpoints at a fixed step ladder into snapshots/<arm>/ as they appear.
It needs no restart of the training jobs and no change to their configs.

It also keeps the newest best-val/mse checkpoint per arm (that file rotates too), pruning
older ones so each arm holds at most one.

Disk: len(LADDER) x ~2.2 GB per arm. With 6 rungs x 4 arms that is ~53 GB.
"""
from __future__ import annotations

import argparse
import re
import shutil
import time
from pathlib import Path

# NOTE: <ROOT>/snapshots is a SYMLINK to
#   /orcd/data/mki_aryeh/001/pablomer/scale_ablation/snapshots
# (the mki_aryeh group allocation: 200 TB, ~1% used, mounted and writable on node4900 and
# node4701 -- both verified). Snapshots therefore cost nothing against the personal POOL
# quota, which is why the ladder below can afford to be dense. Training jobs still write
# their own checkpoints to POOL; only this archiver touches the group filesystem.
ROOT = Path("/orcd/pool/007/pablomer/checkpoints_new/scale_ablation")
ARMS = [1000, 3162, 10000, 31622]
# Every 5k steps, matching train.py's checkpoint_every_n_train_steps=5000 -- so every
# periodic checkpoint the trainer writes gets archived rather than overwritten. 15 rungs x
# 4 arms x ~2.2 GB is ~132 GB, i.e. 0.07% of the group allocation.
# Density matters here: the 1k arm's val/loss turns upward at ~5k steps while its val/mse
# keeps improving past 20k, so pinning down where the ENCODERS actually peak (downstream R2,
# which nothing monitors during training) needs a fine step grid.
LADDER = list(range(5000, 75001, 5000))

STEP_RE = re.compile(r"step=(\d+)\.ckpt$")
BEST_RE = re.compile(r"best-epoch=(\d+)-step=(\d+)\.ckpt$")


def _copy_atomic(src: Path, dst: Path) -> bool:
    """Copy via a temp name so a partially-written file is never mistaken for a snapshot."""
    if dst.exists():
        return False
    tmp = dst.with_suffix(".ckpt.partial")
    try:
        shutil.copy2(src, tmp)
        tmp.replace(dst)
        return True
    except (FileNotFoundError, OSError) as e:
        # The source can vanish mid-copy (that is the whole problem we are solving).
        print(f"  [warn] copy failed {src.name}: {type(e).__name__}: {e}", flush=True)
        tmp.unlink(missing_ok=True)
        return False


def poll_once(root: Path, arms: list[int], ladder: list[int]) -> dict[int, set[int]]:
    """One pass over all arms. Returns {arm: set of ladder steps archived so far}."""
    have: dict[int, set[int]] = {}
    for n in arms:
        out = root / "snapshots" / f"scale_{n}"
        out.mkdir(parents=True, exist_ok=True)
        have[n] = {
            int(m.group(1))
            for p in out.glob("step*.ckpt")
            if (m := re.match(r"step0*(\d+)\.ckpt$", p.name))
        }

        # Rolling periodic checkpoint. The dated run subdir is nondeterministic (each DDP
        # rank makes its own and only one receives files), so glob across all of them.
        for src in (root / "runs" / f"scale_{n}").glob("*/*/checkpoints/latest-step*.ckpt"):
            m = STEP_RE.search(src.name)
            if not m:
                continue
            step = int(m.group(1))
            if step in ladder and step not in have[n]:
                if _copy_atomic(src, out / f"step{step:06d}.ckpt"):
                    print(f"  archived scale_{n} step {step:,}", flush=True)
                    have[n].add(step)

        # Newest best-val/mse checkpoint; prune older ones so each arm keeps one.
        bests = sorted(
            ((int(m.group(2)), p) for p in (root / "best" / f"scale-{n}-75k").glob("best-epoch=*.ckpt")
             if (m := BEST_RE.search(p.name))),
            key=lambda t: t[0],
        )
        if bests:
            step, src = bests[-1]
            dst = out / f"bestmse_step{step:06d}.ckpt"
            if _copy_atomic(src, dst):
                print(f"  archived scale_{n} best-val/mse @ step {step:,}", flush=True)
            for stale in out.glob("bestmse_step*.ckpt"):
                if stale != dst:
                    stale.unlink(missing_ok=True)
                    print(f"  pruned stale best {stale.name} (scale_{n})", flush=True)
    return have


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--arms", type=int, nargs="+", default=ARMS)
    ap.add_argument("--ladder", type=int, nargs="+", default=LADDER)
    ap.add_argument("--interval", type=int, default=300, help="seconds between polls")
    ap.add_argument("--max-hours", type=float, default=23.0)
    ap.add_argument("--once", action="store_true", help="single pass, then exit")
    args = ap.parse_args()

    deadline = time.time() + args.max_hours * 3600
    target = set(args.ladder)
    print(f"[snapshot] arms={args.arms} ladder={args.ladder} interval={args.interval}s", flush=True)

    while True:
        have = poll_once(args.root, args.arms, args.ladder)
        done = all(target <= have[n] for n in args.arms)
        summary = " | ".join(f"scale_{n}:{len(have[n])}/{len(target)}" for n in args.arms)
        print(f"[snapshot] {time.strftime('%H:%M:%S')} {summary}", flush=True)
        if args.once or done or time.time() > deadline:
            print(f"[snapshot] exiting (done={done})", flush=True)
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
