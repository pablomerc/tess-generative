"""
Shared helpers for the redshift-aware anomaly-detection experiment.

Two preprocessing paths — keep them separate:

1) ENCODER path (training parity): raw (5,160,160) float32 ->
     preprocess_image_v2(crop_size=48, survey="hsc")
       -> (img - NORM_DICT["hsc"][0]) / NORM_DICT["hsc"][1]
       -> [:4] (drop y) -> (4,48,48)
   This is the EXACT chain in NeighborsEfficientDataset._preprocess and
   prepare_hsc_downstream.py HSCBinaryDataset.

2) DISPLAY path (paper-figure parity): raw (5,160,160) float32 ->
     preprocess_image_v2(crop_size=160, survey="hsc")  # CenterCrop is a no-op
       -> (5,160,160) float32 -- arcsinh-compressed, NOT normalized
   This matches what is stored in neighbours_v2.h5["images_hsc"] (verified per-band
   medians ~0.005-0.025, p99 0.15-1.2), which is the source the published anomaly
   figures (paper_anomaly_figure.py, visualize_top_anomalies.py) read from.

Channels for RGB: img_chw[:3] = (g, r, i) -> matplotlib (R, G, B). Backwards
from astronomy convention but matches the paper figure verbatim — DO NOT reorder.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Make the repo importable (project root + galaxy_model dir), mirroring the
# sys.path handling in anomaly_detection/encode_latents_ours.py and
# visualization_scripts/regenerate_umap_redshift.py.
#   _HERE = .../galaxy_model/anomaly_detection/redshift_experiment
#   parents[0]=anomaly_detection [1]=galaxy_model [2]=galaxy_images [3]=tess-generative
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_GALAXY_MODEL = _HERE.parents[1]
_PROJECT_ROOT = _HERE.parents[3]
for _p in (str(_PROJECT_ROOT), str(_GALAXY_MODEL)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Force regular hipBLAS instead of hipBLASLt (buggy on MI210). Mirrors the
# other anomaly_detection scripts and the training entry points.
torch.backends.cuda.preferred_blas_library("hipblas")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from torch.utils.data import Dataset  # noqa: E402

from galaxy_images.image_preprocessing import preprocess_image_v2  # noqa: E402
from galaxy_images.galaxy_model.neighbors import NORM_DICT  # noqa: E402

# Default data locations on this cluster.
HSC_DATA_DIR = Path("/work1/jeroenaudenaert/pablomer/data/hsc_downstream")
DEFAULT_CATALOG = HSC_DATA_DIR / "catalog.parquet"
DEFAULT_IMAGES_BIN = HSC_DATA_DIR / "hsc_flux.bin"
DEFAULT_CHECKPOINT = str(_GALAXY_MODEL / "checkpoints" / "base" / "snapshot.ckpt")

HSC_BANDS, HSC_H, HSC_W = 5, 160, 160
HSC_BYTES_PER_GALAXY = HSC_BANDS * HSC_H * HSC_W * 2  # float16


# ---------------------------------------------------------------------------
# Image loading + preprocessing (identical to training)
# ---------------------------------------------------------------------------
def preprocess_hsc(raw_5x160x160: np.ndarray) -> torch.Tensor:
    """Raw (5,160,160) float32 -> preprocessed, normalized (4,48,48) float32 tensor."""
    img = torch.from_numpy(np.asarray(raw_5x160x160, dtype=np.float32))
    img = preprocess_image_v2(img, crop_size=48, survey="hsc")
    mean, std = NORM_DICT["hsc"]
    img = (img - mean) / std
    return img[:4]  # drop y band


class HSCBinDataset(Dataset):
    """Reads hsc_flux.bin records by seek/read (avoids mmap ulimit issues on login nodes).

    `record_idx` are positional row numbers in the binary (== catalog row order).
    One file handle per instance — use num_workers=0.
    """

    def __init__(self, images_bin: str | Path, record_idx: np.ndarray):
        self._images_bin = str(images_bin)
        self.record_idx = np.asarray(record_idx, dtype=np.int64)
        self._file = None

    def __len__(self) -> int:
        return len(self.record_idx)

    def __del__(self):
        if self._file is not None:
            try:
                self._file.close()
            except Exception:
                pass

    def _get_file(self):
        if self._file is None:
            self._file = open(self._images_bin, "rb")
        return self._file

    def __getitem__(self, i):
        f = self._get_file()
        f.seek(int(self.record_idx[i]) * HSC_BYTES_PER_GALAXY)
        buf = f.read(HSC_BYTES_PER_GALAXY)
        raw = np.frombuffer(buf, dtype=np.float16).copy().reshape(HSC_BANDS, HSC_H, HSC_W)
        return preprocess_hsc(raw)


def load_hsc_images(images_bin: str | Path, record_idx: np.ndarray) -> np.ndarray:
    """ENCODER-input tensors. (n, 4, 48, 48) — arcsinh-compressed AND normalized,
    y-band dropped. Use this for the encoder, NOT for display."""
    ds = HSCBinDataset(images_bin, record_idx)
    return np.stack([ds[i].numpy() for i in range(len(ds))], axis=0)


# ---------------------------------------------------------------------------
# Display-only path (matches neighbours_v2.h5["images_hsc"] storage convention)
# ---------------------------------------------------------------------------
def preprocess_hsc_for_display(raw_5x160x160: np.ndarray) -> torch.Tensor:
    """Raw (5,160,160) float32 -> (5,160,160) float32 display tensor.

    Applies `preprocess_image_v2(crop_size=160, survey="hsc")` (band-clamp +
    HSC zeropoint rescale + arcsinh range-compression) but does NOT divide by
    NORM_DICT mean/std — so values match neighbours_v2.h5["images_hsc"] (the
    source the paper figures display from). CenterCrop is a no-op at 160.
    """
    img = torch.from_numpy(np.asarray(raw_5x160x160, dtype=np.float32))
    return preprocess_image_v2(img, crop_size=160, survey="hsc")


class HSCBinDisplayDataset(Dataset):
    """Display-tensor loader. Returns (5,160,160) float32 with the SAME recipe used
    to populate neighbours_v2.h5["images_hsc"]. Use for paper-style RGB tiles.
    """

    def __init__(self, images_bin: str | Path, record_idx: np.ndarray):
        self._images_bin = str(images_bin)
        self.record_idx = np.asarray(record_idx, dtype=np.int64)
        self._file = None

    def __len__(self) -> int:
        return len(self.record_idx)

    def __del__(self):
        if self._file is not None:
            try:
                self._file.close()
            except Exception:
                pass

    def _get_file(self):
        if self._file is None:
            self._file = open(self._images_bin, "rb")
        return self._file

    def __getitem__(self, i):
        f = self._get_file()
        f.seek(int(self.record_idx[i]) * HSC_BYTES_PER_GALAXY)
        buf = f.read(HSC_BYTES_PER_GALAXY)
        raw = np.frombuffer(buf, dtype=np.float16).copy().reshape(HSC_BANDS, HSC_H, HSC_W)
        return preprocess_hsc_for_display(raw)


def load_hsc_images_for_display(images_bin: str | Path, record_idx: np.ndarray) -> np.ndarray:
    """Load a (small) set of HSC images for the paper-style RGB display path.
    Returns (n, 5, 160, 160) float32 — arcsinh-compressed, NOT normalized.
    """
    ds = HSCBinDisplayDataset(images_bin, record_idx)
    return np.stack([ds[i].numpy() for i in range(len(ds))], axis=0)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def load_model(checkpoint_path: str, device: str = "cpu"):
    """Load the trained dual-encoder flow-matching module (eval, no grad)."""
    from double_train_fm_neighbors import ConditionalFlowMatchingModule
    model = ConditionalFlowMatchingModule.load_from_checkpoint(checkpoint_path, map_location="cpu")
    model.eval()
    torch.set_grad_enabled(False)
    return model.to(device)


def encode_physics_mean(model, images: torch.Tensor) -> torch.Tensor:
    """Ours-Physics latent: encoder_1 spatial tokens mean-pooled -> (B, D)."""
    return model.encoder_1(images).mean(dim=1)


def encode_instrument_mean(model, images: torch.Tensor) -> torch.Tensor:
    """Ours-Instrument latent: encoder_2 spatial tokens mean-pooled -> (B, D).

    Symmetric to encode_physics_mean. In the baseline ConditionalFlowMatchingModule
    encoder_2 (the same-instrument encoder) returns spatial tokens (mean_pool=False),
    so mean-pooling over dim=1 yields a (B, D) instrument latent of the same shape as
    the physics latent.
    """
    return model.encoder_2(images).mean(dim=1)


# encoder-name -> encode fn, for --encoder {physics,instrument}
ENCODE_FNS = {"physics": encode_physics_mean, "instrument": encode_instrument_mean}


# ---------------------------------------------------------------------------
# Normalizing-flow anomaly scoring (NSF, score = -log_prob).
#
# Two capacity profiles:
#   "default" — transforms=6, hidden_features=[64,64]  (matches fit_and_score.py)
#   "wide"    — transforms=8, hidden_features=[128,128]  (use for conditional p(x|c))
#
# The wide profile exists because the diagnostic in Fix 4a confirmed the default
# conditional NSF was effectively ignoring its 1-D context (Spearman with the
# unconditional flow = 0.94 despite Ridge R²(z|latent) = 0.59). Wider hyper-nets
# + 2-D [z, z**2] context restore meaningful coupling.
# ---------------------------------------------------------------------------
NSF_PROFILES = {
    "default": dict(transforms=6, hidden_features=[64, 64]),
    "wide":    dict(transforms=8, hidden_features=[128, 128]),
}


def _make_flow(dim: int, context: int, device, profile: str = "default"):
    import zuko
    cfg = NSF_PROFILES[profile]
    return zuko.flows.NSF(features=dim, context=context, **cfg).to(device)


def score_nsf(train_x, all_x, epochs, device_str, lr=1e-3, batch_size=512,
              train_c=None, all_c=None, profile: str = "default",
              cosine_lr: bool = False, return_flow: bool = False):
    """Fit an NSF on train_x (optionally conditioned on train_c) and return -log_prob
    for every row of all_x. If train_c/all_c are given (shape (N, C)), models p(x|c).
    Tracks the best epoch by lowest training loss, as in fit_and_score.py.

    profile: "default" or "wide". `cosine_lr`: use CosineAnnealingLR over `epochs`.
    return_flow: if True, also return the trained flow + standardization stats
                 (latent mean/std used for fit-time scaling; identity here, kept
                 for symmetry with possible future versions). Useful for the
                 context-sensitivity diagnostic.
    """
    from tqdm import tqdm
    device = torch.device(device_str)
    dim = train_x.shape[1]
    conditional = train_c is not None
    ctx_dim = (train_c.shape[1] if conditional else 0)
    flow = _make_flow(dim, context=ctx_dim, device=device, profile=profile)
    optimizer = torch.optim.Adam(flow.parameters(), lr=lr)
    scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
                 if cosine_lr else None)

    train_t = torch.as_tensor(train_x, dtype=torch.float32, device=device)
    n_train = len(train_t)
    train_ct = torch.as_tensor(train_c, dtype=torch.float32, device=device) if conditional else None

    best_loss, best_state = float("inf"), None
    for _ in tqdm(range(epochs), desc=f"  NSF training [{profile}, ctx={ctx_dim}]", leave=False):
        flow.train()
        perm = torch.randperm(n_train, device=device)
        epoch_loss, n_batches = 0.0, 0
        for i in range(0, n_train, batch_size):
            sel = perm[i:i + batch_size]
            batch = train_t[sel]
            optimizer.zero_grad()
            dist = flow(train_ct[sel]) if conditional else flow()
            loss = -dist.log_prob(batch).mean()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        if scheduler is not None:
            scheduler.step()
        avg = epoch_loss / max(n_batches, 1)
        if avg < best_loss:
            best_loss = avg
            best_state = {k: v.clone() for k, v in flow.state_dict().items()}

    if best_state is not None:
        flow.load_state_dict(best_state)
    flow.eval()

    all_t = torch.as_tensor(all_x, dtype=torch.float32, device=device)
    all_ct = torch.as_tensor(all_c, dtype=torch.float32, device=device) if conditional else None
    scores = []
    with torch.no_grad():
        for i in range(0, len(all_t), batch_size):
            dist = flow(all_ct[i:i + batch_size]) if conditional else flow()
            scores.append((-dist.log_prob(all_t[i:i + batch_size])).cpu().numpy())
    scores_arr = np.concatenate(scores, axis=0).astype(np.float32)
    if return_flow:
        return scores_arr, flow
    return scores_arr


def standardize_z(z_raw: np.ndarray, train_mask: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Standardize a 1-D redshift array using train-split statistics. Returns
    (z_standardized, mean, std)."""
    z_raw = np.asarray(z_raw, dtype=np.float64)
    mu = float(z_raw[train_mask].mean())
    sd = float(z_raw[train_mask].std() + 1e-8)
    return ((z_raw - mu) / sd).astype(np.float32), mu, sd


def make_z_context(z_raw: np.ndarray, train_mask: np.ndarray, mode: str = "z") -> np.ndarray:
    """Build the (N, C) context array for the conditional NSF.

    mode="z"    -> (N, 1) standardized z. Same as v1.
    mode="z_z2" -> (N, 2) [z_std, z_std^2]. The diagnostic in Fix 4a confirmed
                   bare-z context was being ignored; squaring gives the hyper-net
                   a direct nonlinear handle without having to discover it itself.
    """
    z_std, _, _ = standardize_z(z_raw, train_mask)
    if mode == "z":
        return z_std.reshape(-1, 1)
    if mode == "z_z2":
        return np.stack([z_std, z_std ** 2], axis=1).astype(np.float32)
    raise ValueError(f"unknown z-context mode: {mode}")


def train_test_split(n, train_frac, seed=42):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_train = int(n * train_frac)
    return idx[:n_train], idx[n_train:]


def top_n_with_percentiles(scores, n):
    """Return (order, top_scores, top_pcts) for the n highest finite scores.
    Percentile = fraction of all finite scores strictly below (0..100).
    """
    scores = np.asarray(scores, dtype=np.float64)
    finite = np.isfinite(scores)
    order = np.argsort(np.where(finite, scores, -np.inf))[::-1][:n]
    sorted_all = np.sort(scores[finite])
    top_scores = scores[order]
    top_pcts = np.array([np.searchsorted(sorted_all, s, side="left") / len(sorted_all) * 100.0
                         for s in top_scores])
    return order, top_scores, top_pcts


def bottom_n_with_percentiles(scores, n):
    """Return (order, bottom_scores, bottom_pcts) for the n LOWEST finite scores
    (most-typical galaxies — the inverse of the anomaly ranking)."""
    scores = np.asarray(scores, dtype=np.float64)
    finite = np.isfinite(scores)
    order = np.argsort(np.where(finite, scores, np.inf))[:n]
    sorted_all = np.sort(scores[finite])
    bot_scores = scores[order]
    bot_pcts = np.array([np.searchsorted(sorted_all, s, side="left") / len(sorted_all) * 100.0
                         for s in bot_scores])
    return order, bot_scores, bot_pcts


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def to_rgb(img_chw, pct_lo=1, pct_hi=99):
    """(C,H,W) float -> (H,W,3) uint8 via per-channel percentile stretch.
    Same as anomaly_detection/visualize_top_anomalies.py:_to_rgb.
    """
    rgb = np.asarray(img_chw[:3], dtype=np.float32)
    lo = np.percentile(rgb, pct_lo, axis=(1, 2), keepdims=True)
    hi = np.percentile(rgb, pct_hi, axis=(1, 2), keepdims=True)
    rgb = np.clip((rgb - lo) / (hi - lo + 1e-8), 0, 1)
    return (rgb.transpose(1, 2, 0) * 255).astype(np.uint8)


def plot_anomaly_grid(images, ranks, z_vals, pcts, title, out_path, n_cols=8):
    """Grid of anomaly tiles. `images` should be DISPLAY tensors (e.g. (n,5,160,160)
    from `load_hsc_images_for_display`) — NOT the encoder-normalized tensors.
    `to_rgb` takes [:3] = g,r,i mapped to (R,G,B) for paper-figure parity.
    """
    n = len(images)
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.7, n_rows * 1.9))
    axes = np.atleast_2d(axes)
    for j in range(n_rows * n_cols):
        ax = axes[j // n_cols, j % n_cols]
        ax.axis("off")
        if j < n:
            ax.imshow(to_rgb(images[j]))
            ax.set_title(f"#{ranks[j]}  z={z_vals[j]:.3f}\np{pcts[j]:.1f}", fontsize=7)
    fig.suptitle(title, fontsize=11, y=0.995)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()
    return out_path


# Property-display metadata: column-name -> (axis label, x-scale).
# Mirrors the audit's recommended axis scales (Fix 3 in the plan).
PROPERTY_AXES = {
    "desi_z":              ("DESI redshift z",                 "linear"),
    "provabgs_logmstar":   ("log10(M* / M$_\\odot$)",          "linear"),
    "provabgs_tage_mw":    ("Mass-wt age (Gyr)",               "log"),
    "provabgs_z_mw":       ("Mass-wt metallicity Z / Z$_\\odot$", "log"),
    "provabgs_avg_sfr":    ("Avg SFR (M$_\\odot$/yr)",         "log"),
}


def _bins_for(values: np.ndarray, scale: str, n: int = 60,
              robust_pct: tuple = (0.5, 99.5)) -> np.ndarray:
    """Bin edges clipped to a robust percentile range so a few extreme outliers
    don't waste the axis (e.g. provabgs_tage_mw has min=0.06 but p1=2.5).
    """
    values = np.asarray(values, dtype=np.float64)
    if scale == "log":
        pos = values[np.isfinite(values) & (values > 0)]
        lo, hi = np.percentile(pos, robust_pct[0]), np.percentile(pos, robust_pct[1])
        return np.logspace(np.log10(lo), np.log10(hi), n)
    finite = values[np.isfinite(values)]
    lo, hi = np.percentile(finite, robust_pct[0]), np.percentile(finite, robust_pct[1])
    return np.linspace(float(lo), float(hi), n)


def _plot_one_property_panel(ax, all_vals, top_vals, label, scale, top_k_label, ax_label_fontsize=9):
    """Single histogram panel: full sample (blue) + top-K rug + faint red twin hist.
    Identical layout to v1's plot_redshift_distribution but parameterized.
    """
    all_vals = np.asarray(all_vals, dtype=np.float64)
    top_vals = np.asarray(top_vals, dtype=np.float64)
    fin_all = np.isfinite(all_vals)
    fin_top = np.isfinite(top_vals)
    if scale == "log":
        fin_all = fin_all & (all_vals > 0)
        fin_top = fin_top & (top_vals > 0)

    bins = _bins_for(all_vals[fin_all], scale)
    ax.hist(all_vals[fin_all], bins=bins, color="#8eb8e8", alpha=0.85,
            edgecolor="black", linewidth=0.3)
    ax.set_xlabel(label, fontsize=ax_label_fontsize)
    ax.set_ylabel(f"All (N={int(fin_all.sum())})", fontsize=ax_label_fontsize - 1)
    if scale == "log":
        ax.set_xscale("log")
    # Clip the visible x-range to the robust bin range so a handful of
    # extreme low/high outliers (e.g. t_age <2 Gyr) don't dominate the axis.
    ax.set_xlim(bins[0], bins[-1])

    # Rug along the bottom + rank labels.
    y_top = ax.get_ylim()[1]
    rug_y = -y_top * 0.05
    ax.set_ylim(rug_y * 1.9, y_top)
    tv = top_vals[fin_top]
    ax.scatter(tv, np.full_like(tv, rug_y), marker="|", s=260, c="red",
               linewidths=1.0, zorder=5)
    for k, sz in enumerate(tv):
        ax.text(sz, rug_y * 1.55, str(k + 1), fontsize=5, ha="center", va="top", color="darkred")

    # Twin axis: top-K histogram in red.
    ax2 = ax.twinx()
    ax2.hist(tv, bins=bins, color="red", alpha=0.5, edgecolor="darkred", linewidth=0.3)
    ax2.set_ylabel(f"{top_k_label} (N={int(fin_top.sum())})", color="red", fontsize=ax_label_fontsize - 1)
    ax2.tick_params(axis="y", colors="red", labelsize=7)

    if scale == "log":
        med_all = float(np.exp(np.log(all_vals[fin_all]).mean())) if fin_all.any() else np.nan  # geom mean
        med_top = float(np.exp(np.log(tv).mean())) if fin_top.any() else np.nan
        ax.set_title(f"geom-mean all={med_all:.3g} | top={med_top:.3g}", fontsize=8)
    else:
        ax.set_title(f"median all={np.median(all_vals[fin_all]):.3g} | "
                     f"top={np.median(tv):.3g}" if fin_top.any() else "no top points",
                     fontsize=8)


def plot_property_distributions(props: dict, top_idx: np.ndarray, title: str,
                                out_path, top_k_label: str = "top anomalies"):
    """5-panel figure: one panel per physics property, each showing
    full-sample histogram + top-K rug + faint red top-K twin histogram.

    `props` keys must be a subset of PROPERTY_AXES; each value is a length-N array
    aligned to the working set. `top_idx` are the 0..N-1 indices of the top-K
    anomalies (descending score order). Skips any column where every value is NaN.
    """
    keys = [k for k in PROPERTY_AXES if k in props]
    n_panels = len(keys)
    fig, axes = plt.subplots(1, n_panels, figsize=(4.5 * n_panels, 4.2))
    axes = np.atleast_1d(axes)
    for ax, k in zip(axes, keys):
        label, scale = PROPERTY_AXES[k]
        all_vals = np.asarray(props[k])
        top_vals = all_vals[top_idx]
        _plot_one_property_panel(ax, all_vals, top_vals, label, scale, top_k_label)
    fig.suptitle(title, fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close()
    return out_path


# ---------------------------------------------------------------------------
# Discord
# ---------------------------------------------------------------------------
def discord_notify(webhook_url, message, file_path=None):
    """Thin wrapper over visualization_scripts/discord_notify.notify (never raises)."""
    try:
        from visualization_scripts.discord_notify import notify
    except Exception:
        sys.path.insert(0, str(_GALAXY_MODEL / "visualization_scripts"))
        from discord_notify import notify
    notify(webhook_url, message, file_path=file_path)
