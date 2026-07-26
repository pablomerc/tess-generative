#!/usr/bin/env python3
"""Render Discord-ready paired-image galleries from a frozen quality table."""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
import uuid
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from common import require_file

HSC_SHAPE = (5, 160, 160)
LEGACY_SHAPE = (4, 160, 160)


def _binary_rows(path: Path, shape: tuple[int, int, int]) -> int:
    require_file(path, "image binary")
    record = int(np.prod(shape)) * np.dtype(np.float16).itemsize
    size = path.stat().st_size
    if size % record:
        # Match prepare_combined.py: complete records begin at byte zero and any
        # incomplete trailer is ignored. Every requested index is still checked.
        print(
            f"warning: {path} has {size % record} trailing bytes; ignoring trailer",
            file=sys.stderr,
        )
    return size // record


def read_record(path: Path, index: int, shape: tuple[int, int, int]) -> np.ndarray:
    record = int(np.prod(shape)) * np.dtype(np.float16).itemsize
    n = _binary_rows(path, shape)
    if index < 0 or index >= n:
        raise IndexError(f"index {index} outside {path.name} row range [0, {n})")
    with open(path, "rb") as fh:
        fh.seek(index * record)
        data = fh.read(record)
    if len(data) != record:
        raise IOError(f"short read for {path} row {index}: {len(data)} != {record}")
    return np.frombuffer(data, dtype=np.float16).astype(np.float32).reshape(shape)


def center_crop(x: np.ndarray, size: int) -> np.ndarray:
    h, w = x.shape[-2:]
    if size > min(h, w):
        raise ValueError(f"crop {size} exceeds image shape {x.shape}")
    y0, x0 = (h - size) // 2, (w - size) // 2
    return x[:, y0 : y0 + size, x0 : x0 + size]


def legacy_fov_zoom(x: np.ndarray, factor: float = 0.64) -> np.ndarray:
    """Match the model's Legacy field-of-view zoom, without model normalization."""
    out_size = x.shape[-1]
    inner = max(2, round(factor * out_size))
    cropped = center_crop(x, inner)
    return np.stack(
        [
            np.asarray(
                Image.fromarray(ch, mode="F").resize((out_size, out_size), Image.Resampling.BILINEAR),
                dtype=np.float32,
            )
            for ch in cropped
        ]
    )


def robust_asinh_rgb(x: np.ndarray, stretch: float = 8.0) -> np.ndarray:
    """Independent background subtraction and robust asinh scale for one survey."""
    # HSC/Legacy channel order is griz; display i/r/g as R/G/B.
    channels = x[[2, 1, 0]].astype(np.float64)
    rgb = np.empty_like(channels)
    for j, channel in enumerate(channels):
        border = np.concatenate(
            (channel[:4].ravel(), channel[-4:].ravel(), channel[:, :4].ravel(), channel[:, -4:].ravel())
        )
        background = np.nanmedian(border)
        centered = channel - background
        positive = centered[np.isfinite(centered) & (centered > 0)]
        scale = np.nanpercentile(positive, 99.5) if positive.size else np.nan
        if not np.isfinite(scale) or scale <= 0:
            mad = np.nanmedian(np.abs(centered - np.nanmedian(centered)))
            scale = max(6.0 * mad, 1e-6)
        rgb[j] = np.arcsinh(stretch * centered / scale) / np.arcsinh(stretch)
    rgb = np.nan_to_num(rgb, nan=0.0, posinf=1.0, neginf=0.0)
    displayed = np.clip(np.moveaxis(rgb, 0, -1), 0, 1)
    # A final conversion also covers unusual numpy builds where assignment from
    # a non-native floating dtype can preserve a signaling NaN.
    return np.nan_to_num(displayed, nan=0.0, posinf=1.0, neginf=0.0)


def select_strata(df: pd.DataFrame, n: int, seed: int) -> dict[str, pd.DataFrame]:
    valid = df[df["joint_score"].notna()].sort_values("joint_score").reset_index(drop=True)
    if len(valid) < 3:
        raise ValueError("need at least three finite joint_score rows for galleries")
    q20, q40, q60, q80 = valid["joint_score"].quantile([0.2, 0.4, 0.6, 0.8])
    pools = {
        "top": valid[valid["joint_score"] >= q80],
        "middle": valid[(valid["joint_score"] >= q40) & (valid["joint_score"] <= q60)],
        "bottom": valid[valid["joint_score"] <= q20],
    }
    rng = np.random.default_rng(seed)
    return {
        name: pool.iloc[np.sort(rng.choice(len(pool), min(n, len(pool)), replace=False))]
        for name, pool in pools.items()
    }


def render_sheet(
    rows: pd.DataFrame,
    hsc_bin: Path,
    legacy_bin: Path,
    out: Path,
    crop_size: int,
    title: str,
) -> None:
    fig, axes = plt.subplots(len(rows), 2, figsize=(6.4, 2.7 * len(rows)), squeeze=False)
    fig.suptitle(title, fontsize=13)
    for i, (_, row) in enumerate(rows.iterrows()):
        hsc = center_crop(read_record(hsc_bin, int(row.hsc_image_idx), HSC_SHAPE), crop_size)
        legacy = center_crop(read_record(legacy_bin, int(row.legacy_image_idx), LEGACY_SHAPE), crop_size)
        legacy = legacy_fov_zoom(legacy)
        axes[i, 0].imshow(robust_asinh_rgb(hsc))
        axes[i, 1].imshow(robust_asinh_rgb(legacy))
        axes[i, 0].set_title(
            f"HSC  id={row.hsc_object_id}\nres={row.resolution_gap:+.3g}  snr={row.snr_gap:+.3g}",
            fontsize=8,
        )
        axes[i, 1].set_title(
            f"Legacy  id={row.legacy_object_id}\njoint={row.joint_score:.3f}",
            fontsize=8,
        )
        for ax in axes[i]:
            ax.axis("off")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(out, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def render_summary(df: pd.DataFrame, out: Path) -> None:
    valid = df[df["joint_score"].notna()]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    sc = axes[0].scatter(
        valid["resolution_gap"],
        valid["snr_gap"],
        c=valid["joint_score"],
        s=9,
        alpha=0.65,
        cmap="viridis",
    )
    axes[0].set(xlabel="resolution gap (HSC − Legacy)", ylabel="log-SNR gap (HSC − Legacy)")
    fig.colorbar(sc, ax=axes[0], label="joint score")
    axes[1].hist(valid["joint_score"], bins=30, color="#4c78a8", alpha=0.9)
    axes[1].set(xlabel="joint score", ylabel="objects")
    fig.suptitle(f"Paired quality ranking — {valid.metric_version.iloc[0]}")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def post_discord(webhook: str, image: Path, message: str) -> None:
    boundary = f"----quality-gallery-{uuid.uuid4().hex}"
    payload = json.dumps({"content": message}).encode()
    body = (
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"payload_json\"\r\n"
        "Content-Type: application/json\r\n\r\n"
    ).encode() + payload + b"\r\n"
    body += (
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; "
        f"filename=\"{image.name}\"\r\nContent-Type: image/png\r\n\r\n"
    ).encode() + image.read_bytes() + f"\r\n--{boundary}--\r\n".encode()
    request = urllib.request.Request(
        webhook,
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        if response.status >= 300:
            raise RuntimeError(f"Discord returned HTTP {response.status}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quality-table", required=True, type=Path)
    ap.add_argument("--hsc-bin", required=True, type=Path)
    ap.add_argument("--legacy-bin", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--n-per-stratum", type=int, default=12)
    ap.add_argument("--crop-size", type=int, default=48)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--webhook", default=os.environ.get("DISCORD_WEBHOOK", ""))
    args = ap.parse_args()
    for p, label in (
        (args.quality_table, "quality table"),
        (args.hsc_bin, "HSC binary"),
        (args.legacy_bin, "Legacy binary"),
    ):
        require_file(p, label)
    df = pd.read_parquet(args.quality_table)
    needed = {
        "hsc_image_idx",
        "legacy_image_idx",
        "hsc_object_id",
        "legacy_object_id",
        "resolution_gap",
        "snr_gap",
        "joint_score",
        "metric_version",
    }
    missing = needed - set(df)
    if missing:
        raise ValueError(f"quality table missing columns: {sorted(missing)}")
    versions = df["metric_version"].dropna().unique()
    if len(versions) != 1:
        raise ValueError(f"quality table must have one metric_version, found {versions}")
    _binary_rows(args.hsc_bin, HSC_SHAPE)
    _binary_rows(args.legacy_bin, LEGACY_SHAPE)

    files = []
    for name, rows in select_strata(df, args.n_per_stratum, args.seed).items():
        out = args.out_dir / f"quality_{name}.png"
        render_sheet(rows, args.hsc_bin, args.legacy_bin, out, args.crop_size, f"{name.title()} quality stratum")
        files.append(out)
        print(f"wrote {out}")
    summary = args.out_dir / "quality_summary.png"
    render_summary(df, summary)
    files.append(summary)
    print(f"wrote {summary}")
    if args.webhook:
        for path in files:
            post_discord(args.webhook, path, f"quality gallery: {path.stem} ({versions[0]})")
        print(f"posted {len(files)} PNGs to Discord")
    else:
        print("no --webhook or DISCORD_WEBHOOK; skipped Discord upload")


if __name__ == "__main__":
    main()
