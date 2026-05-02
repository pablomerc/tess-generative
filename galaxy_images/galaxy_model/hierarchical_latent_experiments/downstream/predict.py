"""Wrapper around contrastive predict_all_contrastive.py — neighbors only.

Reuses train/eval machinery (MLPRegressor, LitRegressor, evaluate_per_target, ...) verbatim.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import torch

if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "preferred_blas_library"):
    torch.backends.cuda.preferred_blas_library("hipblas")

_THIS = Path(__file__).resolve()
_PKG_ROOT = _THIS.parents[4]
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from galaxy_images.galaxy_model.contrastive_baseline.downstream_evaluation.predict_all_contrastive import (
    TASKS, plot_objective, run_task, save_results_csv,
)
from galaxy_images.galaxy_model.hierarchical_latent_experiments.common.discord_utils import (
    post_image, post_text,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True)
    ap.add_argument("--suffix", required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-gpu", action="store_true")
    ap.add_argument("--no-discord", action="store_true")
    args = ap.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if not args.no_discord:
        post_text(f"🟢 [downstream/predict/{args.variant}] starting (suffix={args.suffix})")

    t0 = time.time()
    h5_path = output_dir / f"downstream_neighbors_{args.suffix}.h5"
    if not h5_path.exists():
        msg = f"❌ [downstream/predict/{args.variant}] missing {h5_path.name}"
        if not args.no_discord:
            post_text(msg)
        raise FileNotFoundError(h5_path)

    use_gpu = (not args.no_gpu) and torch.cuda.is_available()
    print(f"GPU enabled: {use_gpu}; H5: {h5_path}")

    all_results = []
    for objective, task_name, h5_stem, default_use_embedding, targets in TASKS:
        if h5_stem != "neighbors":
            continue  # cluster only has neighbors data
        if not targets:
            continue
        print(f"\n--- {objective} :: {task_name} ({len(targets)} targets) ---")
        try:
            res, names = run_task(task_name, h5_stem, default_use_embedding, targets,
                                  h5_path, args.seed, use_gpu)
        except Exception as exc:
            print(f"[skip] {task_name}: {exc}")
            if not args.no_discord:
                post_text(f"⚠️ [downstream/predict/{args.variant}] {task_name} skipped: {exc}")
            continue
        all_results.append((objective, task_name, res, names))

    csv_path = output_dir / f"predict_{args.variant}_{args.suffix}.csv"
    save_results_csv(all_results, csv_path)

    df = pd.read_csv(csv_path)
    plot_paths = []
    for objective in sorted(df["objective"].unique()):
        df_obj = df[df["objective"] == objective].copy()
        plot_path = output_dir / f"predict_{args.variant}_{args.suffix}_{objective}.png"
        plot_objective(df_obj, objective, plot_path)
        plot_paths.append(plot_path)

    elapsed = time.time() - t0
    print(f"done in {elapsed:.1f}s; csv={csv_path}")

    if not args.no_discord:
        try:
            r2_phys = pd.to_numeric(df.get("r2_physics", pd.Series(dtype=float)), errors="coerce").mean()
            r2_inst = pd.to_numeric(df.get("r2_instrument", pd.Series(dtype=float)), errors="coerce").mean()
            r2_untr = pd.to_numeric(df.get("r2_untrained", pd.Series(dtype=float)), errors="coerce").mean()
            summary = (f"variant={args.variant}  mean R²: "
                       f"physics={r2_phys:.3f}  instrument={r2_inst:.3f}  untrained={r2_untr:.3f}")
        except Exception:
            summary = ""
        for p in plot_paths:
            post_image(p, message=f"📊 [downstream/{args.variant}] {p.name}")
        post_text(f"✅ [downstream/predict/{args.variant}] done in {elapsed:.0f}s — {summary}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        try:
            from galaxy_images.galaxy_model.hierarchical_latent_experiments.common.discord_utils import post_text as _pt
            _pt(f"❌ [downstream/predict] FAILED: {type(exc).__name__}: {exc}")
        except Exception:
            pass
        raise
