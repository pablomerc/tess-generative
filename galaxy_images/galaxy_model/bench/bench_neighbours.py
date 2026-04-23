"""
Bottleneck benchmark for train_neighbours.

Runs three isolated timing phases on a single shard so you don't need
the full VDS:

  Phase 1 – RAM load      : how long does NeighborsPrecomputedDataset.__init__ take?
  Phase 2 – DataLoader    : steady-state batch throughput (samples/s, ms/batch)
  Phase 3 – Train step    : forward + backward + optimizer step (ms/step)

At the end it prints a short diagnosis and saves a Markdown report to
  <output_dir>/bench_<timestamp>.md

Usage (interactive):
  python galaxy_images/galaxy_model/bench/bench_neighbours.py

Override knobs via CLI:
  python galaxy_images/galaxy_model/bench/bench_neighbours.py \
      --h5 /path/to/shard.h5 \
      --batch-size 64 \
      --num-workers 0 \
      --warmup-batches 5 \
      --bench-batches 20 \
      --train-steps 10 \
      --output-dir galaxy_images/galaxy_model/bench/results
"""

from __future__ import annotations

import argparse
import datetime
import io
import os
import sys
import time
from pathlib import Path

import torch
# Force regular hipBLAS instead of hipBLASLt — hipBLASLt is buggy on MI210
# for certain matrix shapes and causes HIPBLAS_STATUS_INVALID_VALUE at runtime.
torch.backends.cuda.preferred_blas_library("hipblas")
from torch.utils.data import DataLoader

# ── project root on path ─────────────────────────────────────────────────────
_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
if _root not in sys.path:
    sys.path.insert(0, _root)

from galaxy_images.galaxy_model.neighbors import (
    NeighborsPrecomputedDataset,
    simple_collate,
)

# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_SHARD = "/work1/jeroenaudenaert/pablomer/data/neighbor_batches/neighbors_shard_0000.h5"
DEFAULT_CONFIG = os.path.join(os.path.dirname(__file__), "../configs/neighbors_default.json")
DEFAULT_OUTPUT = os.path.join(os.path.dirname(__file__), "results")


# ─── Tee: write to stdout AND collect lines for the report ───────────────────

class _Tee:
    """Wraps sys.stdout so every print() also accumulates into self.lines."""
    def __init__(self):
        self._orig = sys.stdout
        self.lines: list[str] = []

    def __enter__(self):
        sys.stdout = self
        return self

    def __exit__(self, *_):
        sys.stdout = self._orig

    def write(self, text):
        self._orig.write(text)
        self.lines.append(text)

    def flush(self):
        self._orig.flush()

    def getvalue(self) -> str:
        return "".join(self.lines)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _hline(char="─", width=70):
    print(char * width)


def _section(title: str):
    _hline()
    print(f"  {title}")
    _hline()


def parse_args():
    p = argparse.ArgumentParser(description="Bottleneck benchmark for train_neighbours")
    p.add_argument("--h5", default=DEFAULT_SHARD,
                   help="Path to a single precomputed shard (.h5)")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=0,
                   help="DataLoader workers (0 = main process)")
    p.add_argument("--warmup-batches", type=int, default=5,
                   help="Batches to skip before timing DataLoader")
    p.add_argument("--bench-batches", type=int, default=20,
                   help="Batches to time for DataLoader phase")
    p.add_argument("--train-steps", type=int, default=10,
                   help="Forward+backward steps for model phase")
    p.add_argument("--no-gpu", action="store_true",
                   help="Force CPU even if GPU is available")
    p.add_argument("--config", default=DEFAULT_CONFIG,
                   help="Config JSON (used only to build the model)")
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT,
                   help="Directory where the .md report is saved")
    return p.parse_args()


# ─── Phase 1: RAM Load ───────────────────────────────────────────────────────

def phase_ram_load(h5_path: str):
    _section("Phase 1 – RAM load")
    t0 = time.perf_counter()
    dataset = NeighborsPrecomputedDataset(h5_path)
    elapsed = time.perf_counter() - t0
    print(f"\n  → RAM load time : {elapsed:.2f} s")
    print(f"  → Dataset size  : {len(dataset):,} samples")
    return dataset, elapsed


# ─── Phase 2: DataLoader throughput ──────────────────────────────────────────

def phase_dataloader(dataset, batch_size, num_workers, warmup, n_bench):
    _section("Phase 2 – DataLoader throughput")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=simple_collate,
        persistent_workers=(num_workers > 0),
        pin_memory=(num_workers > 0),
        drop_last=True,
    )

    total_batches = len(loader)
    n_bench = min(n_bench, max(1, total_batches - warmup))
    print(f"  Loader has {total_batches} batches  |  warmup={warmup}  |  timing={n_bench}")

    it = iter(loader)
    print("  Warming up...", end=" ", flush=True)
    for _ in range(min(warmup, total_batches)):
        next(it)
    print("done")

    t0 = time.perf_counter()
    counted = 0
    for _ in range(n_bench):
        try:
            next(it)
            counted += 1
        except StopIteration:
            break
    elapsed = time.perf_counter() - t0

    if counted == 0:
        print("  ✗  Not enough batches to benchmark DataLoader")
        return float("inf"), float("inf")

    ms_per_batch = elapsed / counted * 1000
    samples_per_sec = counted * batch_size / elapsed
    print(f"\n  → {counted} batches in {elapsed:.3f} s")
    print(f"  → {ms_per_batch:.1f} ms / batch   |   {samples_per_sec:.0f} samples / s")
    return ms_per_batch, samples_per_sec


# ─── Phase 3: Model train step ───────────────────────────────────────────────

def _build_model_and_optimizer(config_path: str, device: torch.device):
    from dataclasses import asdict
    from galaxy_images.galaxy_model.config import load_experiment_config
    from galaxy_images.galaxy_model.variants import (
        filter_supported_model_kwargs,
        get_variant,
    )

    config = load_experiment_config(config_path, [])
    variant = get_variant(config.run.variant)
    model_kwargs = asdict(config.model)
    model_kwargs["channel_mult"] = tuple(model_kwargs["channel_mult"])
    model_kwargs.update(variant.model_overrides)
    model_kwargs = filter_supported_model_kwargs(variant.model_cls, model_kwargs)
    model = variant.model_cls(**model_kwargs).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.model.lr)
    return model, optimizer


def phase_train_step(dataset, batch_size, n_steps, device, config_path):
    _section("Phase 3 – Model forward + backward")

    print("  Building model...", end=" ", flush=True)
    try:
        model, optimizer = _build_model_and_optimizer(config_path, device)
    except Exception as e:
        print(f"\n  ✗  Could not build model: {e}")
        print("  (Skipping Phase 3 — supply a valid --config to enable)")
        return float("nan"), float("nan")
    print("done")

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Model params: {n_params:.1f} M")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=simple_collate,
        drop_last=True,
    )
    it = iter(loader)
    model.train()

    def next_batch():
        try:
            return next(it)
        except StopIteration:
            return None

    def run_step(batch):
        targets, samegals, sameins, masks, meta = batch
        targets  = targets.to(device)
        samegals = samegals.to(device)
        sameins  = sameins.to(device)
        masks    = masks.to(device)
        loss = model.compute_loss((targets, samegals, sameins, masks, meta))
        if isinstance(loss, dict):
            loss = loss["loss"]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.item()

    # warmup step
    print("  Warmup step...", end=" ", flush=True)
    batch = next_batch()
    if batch is None:
        print("\n  ✗  No batches available")
        return float("nan"), float("nan")
    try:
        run_step(batch)
    except Exception as e:
        print(f"\n  ✗  Forward/backward failed: {e}")
        return float("nan"), float("nan")
    if device.type == "cuda":
        torch.cuda.synchronize()
    print("done")

    step_times = []
    for _ in range(n_steps):
        batch = next_batch()
        if batch is None:
            break
        t0 = time.perf_counter()
        run_step(batch)
        if device.type == "cuda":
            torch.cuda.synchronize()
        step_times.append((time.perf_counter() - t0) * 1000)

    if not step_times:
        print("  ✗  No steps measured")
        return float("nan"), float("nan")

    avg_ms = sum(step_times) / len(step_times)
    throughput = batch_size / avg_ms * 1000
    print(f"\n  → {len(step_times)} steps measured")
    print(f"  → {avg_ms:.1f} ms / step  (min {min(step_times):.1f}  max {max(step_times):.1f})")
    print(f"  → Estimated throughput: {throughput:.0f} samples / s")
    return avg_ms, throughput


# ─── Diagnosis ───────────────────────────────────────────────────────────────

def diagnose(ram_s, dl_ms, dl_sps, step_ms, step_sps, batch_size):
    _section("Diagnosis")

    if ram_s > 60:
        print(f"  ⚠  RAM load took {ram_s:.1f} s for ONE shard.")
        print("     The full VDS spans many shards — loading will be proportionally longer.")
        print("     Consider streaming from disk (num_workers > 0) or faster storage.")
    else:
        print(f"  ✓  RAM load is fast ({ram_s:.1f} s for this shard).")

    nan = float("nan")
    if dl_ms != float("inf") and step_ms == step_ms:  # step_ms is not nan
        if dl_ms > step_ms * 0.5:
            print(f"\n  ⚠  DataLoader ({dl_ms:.1f} ms/batch) is slow relative to compute "
                  f"({step_ms:.1f} ms/step).")
            print("     Try increasing --num-workers.")
        else:
            print(f"\n  ✓  DataLoader ({dl_ms:.1f} ms/batch) is NOT the bottleneck.")
            print(f"     Compute (GPU) dominates at {step_ms:.1f} ms/step.")

        steps_per_hour = 3600 / (step_ms / 1000)
        samples_per_hour_M = steps_per_hour * batch_size / 1e6
        print(f"\n  Estimated training speed:")
        print(f"    {steps_per_hour:,.0f} steps / hour")
        print(f"    {samples_per_hour_M:.2f} M samples / hour")

    _hline()


# ─── Markdown report ─────────────────────────────────────────────────────────

def save_report(
    output_dir: str,
    raw_output: str,
    args,
    device: torch.device,
    ram_s: float,
    dl_ms: float,
    dl_sps: float,
    step_ms: float,
    step_sps: float,
    timestamp: str,
):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    report_path = out / f"bench_{timestamp}.md"

    def _fmt(v, unit="", fmt=".1f"):
        if v != v:  # nan
            return "n/a (model build failed)"
        if v == float("inf"):
            return "n/a (not enough batches)"
        return f"{v:{fmt}} {unit}".strip()

    gpu_line = (
        f"{torch.cuda.get_device_name(0)}" if device.type == "cuda" else "CPU only"
    )
    steps_per_hour = (3600 / (step_ms / 1000)) if (step_ms == step_ms and step_ms > 0) else float("nan")

    md = f"""# Neighbour Training Benchmark — {timestamp}

## Configuration

| Setting | Value |
|---|---|
| Shard | `{args.h5}` |
| Batch size | {args.batch_size} |
| DataLoader workers | {args.num_workers} |
| Warmup batches | {args.warmup_batches} |
| Bench batches | {args.bench_batches} |
| Train steps measured | {args.train_steps} |
| Device | {device} ({gpu_line}) |

---

## Results

| Phase | Metric | Value |
|---|---|---|
| **1 – RAM load** | Time for shard → RAM | {_fmt(ram_s, 's')} |
| **2 – DataLoader** | Time per batch | {_fmt(dl_ms, 'ms')} |
| **2 – DataLoader** | Throughput | {_fmt(dl_sps, 'samples/s', '.0f')} |
| **3 – Train step** | Time per step (fwd+bwd+opt) | {_fmt(step_ms, 'ms')} |
| **3 – Train step** | Throughput | {_fmt(step_sps, 'samples/s', '.0f')} |
| **3 – Train step** | Estimated speed | {_fmt(steps_per_hour, 'steps/hour', '.0f')} |

---

## Full console output

```
{raw_output.strip()}
```
"""

    report_path.write_text(md, encoding="utf-8")
    return report_path


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    device = torch.device("cpu")
    if not args.no_gpu and torch.cuda.is_available():
        device = torch.device("cuda")

    print(f"\n  Device    : {device}")
    if device.type == "cuda":
        print(f"  GPU       : {torch.cuda.get_device_name(0)}")
    print(f"  Shard     : {args.h5}")
    print(f"  Timestamp : {timestamp}\n")

    with _Tee() as tee:
        dataset, ram_s      = phase_ram_load(args.h5)
        dl_ms, dl_sps       = phase_dataloader(
            dataset, args.batch_size, args.num_workers,
            args.warmup_batches, args.bench_batches,
        )
        step_ms, step_sps   = phase_train_step(
            dataset, args.batch_size, args.train_steps, device, args.config,
        )
        diagnose(ram_s, dl_ms, dl_sps, step_ms, step_sps, args.batch_size)

    report_path = save_report(
        args.output_dir, tee.getvalue(), args, device,
        ram_s, dl_ms, dl_sps, step_ms, step_sps, timestamp,
    )
    print(f"\n  Report saved → {report_path}\n")


if __name__ == "__main__":
    main()
