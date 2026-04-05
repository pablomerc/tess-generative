"""
Stage 2 evaluation: apply the trained ResNet to flow-matching generated images.

Pipeline
--------
1. Load a neighbors shard (not used during ResNet training).
2. Filter to HSC-target galaxies (meta_survey == 'hsc').
3. Generate 1 posterior sample per galaxy from the flow-matching model.
4. Crossmatch via object ID to get ground-truth (SHAPE_E1, SHAPE_E2):
     shard['meta_idx'][i]
       → neighbours_v2.h5['object_id_legacy'][meta_idx]
       → lookup dict built from resnet_data.h5 (ls_object_id → e1, e2)
5. Run trained ResNet on:
     a) real targets  → predicted ellipticities
     b) generated images → predicted ellipticities
6. Plot and compare predictions vs. ground truth for real vs. generated.

Key constraint: NEVER use positional alignment. Always join by object ID.

Usage:
  python evaluate_generated.py \\
      --shard  /path/to/neighbors_shard_NNNN.h5 \\
      --checkpoint /path/to/flow_matching.ckpt \\
      [--n-galaxies 200] [--steps 250] [--no-compile]
"""

import argparse
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
from scipy import stats

_here = Path(__file__).resolve().parent
_galaxy_model = _here.parent
sys.path.insert(0, str(_galaxy_model))

RESNET_CKPT   = _here / "resnet_best.pth"
RESNET_DATA   = _here / "resnet_data.h5"
NEIGHBOURS_H5 = Path("/data/vision/billf/scratch/pablomer/data/neighbours_v2.h5")
OUTPUT_DIR    = _here / "outputs"

NORM_MEAN = 0.022
NORM_STD  = 0.05

DEFAULT_CHECKPOINT = (
    "/data/vision/billf/scratch/pablomer/projects/tess-generative/"
    "galaxy-flow-matching-neighbours/g2g9kvr4/checkpoints/"
    "latest-step=step=75000.ckpt"
)


# ---------------------------------------------------------------------------
# ResNet (same architecture as train.py — must match)
# ---------------------------------------------------------------------------

def build_resnet18() -> nn.Module:
    import torchvision.models as tv_models
    model = tv_models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc    = nn.Linear(512, 2)
    return model


def load_resnet(ckpt_path: Path, device: torch.device) -> nn.Module:
    print(f"Loading ResNet from {ckpt_path}")
    ckpt  = torch.load(ckpt_path, map_location=device)
    model = build_resnet18()
    model.load_state_dict(ckpt["model_state"])
    model.eval().to(device)
    return model


# ---------------------------------------------------------------------------
# Flow-matching model
# ---------------------------------------------------------------------------

def load_flow_model(checkpoint_path: str, device: torch.device):
    from double_train_fm_neighbors import ConditionalFlowMatchingModule
    print(f"Loading flow-matching checkpoint: {checkpoint_path}")
    model = ConditionalFlowMatchingModule.load_from_checkpoint(
        checkpoint_path, map_location=device
    )
    model.eval().to(device)
    torch.set_grad_enabled(False)
    return model


# ---------------------------------------------------------------------------
# Crossmatch lookup
# ---------------------------------------------------------------------------

def build_id_to_ellipticity_lookup(resnet_data_path: Path) -> dict:
    """
    Build {legacy_object_id_bytes → (e1, e2)} from resnet_data.h5.
    This is the only authoritative source for SHAPE_E2.
    """
    print(f"Building ID → ellipticity lookup from {resnet_data_path} …")
    lookup = {}
    with h5py.File(resnet_data_path, "r") as f:
        ids = f["ls_object_id"][:]    # (N,) bytes
        e1  = f["shape_e1"][:]        # (N,) float32
        e2  = f["shape_e2"][:]        # (N,) float32
    for i in range(len(ids)):
        lookup[ids[i]] = (float(e1[i]), float(e2[i]))
    print(f"  Lookup built: {len(lookup):,} entries")
    return lookup


def load_legacy_object_ids(neighbours_h5_path: Path) -> np.ndarray:
    """Load object_id_legacy from neighbours_v2.h5 → array of bytes."""
    print(f"Loading object_id_legacy from {neighbours_h5_path} …")
    with h5py.File(neighbours_h5_path, "r") as f:
        ids = f["object_id_legacy"][:]   # (N,) bytes or int
    return ids


def resolve_ellipticities(meta_idx_arr, legacy_ids, lookup) -> tuple:
    """
    For each entry in meta_idx_arr:
      1. legacy_ids[meta_idx] → legacy object ID
      2. lookup[id] → (e1, e2)
    Returns (e1_arr, e2_arr, valid_mask) — valid_mask is False when ID not found.
    """
    e1_out   = np.full(len(meta_idx_arr), np.nan, dtype=np.float32)
    e2_out   = np.full(len(meta_idx_arr), np.nan, dtype=np.float32)
    valid    = np.zeros(len(meta_idx_arr), dtype=bool)

    for i, idx in enumerate(meta_idx_arr):
        raw_id = legacy_ids[idx]
        # Normalise to bytes
        if isinstance(raw_id, (int, np.integer)):
            key = str(int(raw_id)).encode()
        elif isinstance(raw_id, str):
            key = raw_id.encode()
        else:
            key = bytes(raw_id)

        if key in lookup:
            e1_out[i], e2_out[i] = lookup[key]
            valid[i] = True

    n_found = valid.sum()
    print(f"  Resolved {n_found}/{len(meta_idx_arr)} IDs in lookup")
    return e1_out, e2_out, valid


# ---------------------------------------------------------------------------
# Shard loading & generation
# ---------------------------------------------------------------------------

def load_shard(shard_path: str):
    print(f"Loading shard: {shard_path}")
    with h5py.File(shard_path, "r") as f:
        targets  = f["targets"][:]           # (N, 4, 48, 48)
        samegals = f["samegals"][:]          # (N, 4, 48, 48)
        sameins  = f["sameins"][:]           # (N, k, 4, 48, 48)
        masks    = f["neighbor_masks"][:]    # (N, k)
        surveys  = f["meta_survey"][:]       # (N,) bytes
        meta_idx = f["meta_idx"][:]          # (N,) int64
    print(f"  Loaded {len(targets)} entries")
    return targets, samegals, sameins, masks, surveys, meta_idx


def generate_one_per_galaxy(model, targets, samegals, sameins, masks, indices,
                             num_steps: int, device: torch.device,
                             compile_model: bool = True):
    """
    Generate 1 sample per galaxy.
    indices: which rows from the shard to process.
    Returns (N, 4, 48, 48) float32 array.
    """
    if compile_model:
        print("Compiling velocity model …")
        model.velocity_model = torch.compile(model.velocity_model)

    N = len(indices)
    generated = np.zeros((N, 4, 48, 48), dtype=np.float32)
    print(f"Generating {N} samples ({num_steps} Euler steps each) …")
    t0 = time.time()

    for out_i, src_i in enumerate(indices):
        sg = torch.from_numpy(samegals[src_i]).unsqueeze(0).to(device)   # (1, 4, 48, 48)
        si = torch.from_numpy(sameins[src_i]).unsqueeze(0).to(device)    # (1, k, 4, 48, 48)
        mk = torch.from_numpy(masks[src_i]).unsqueeze(0).to(device)      # (1, k)

        with torch.no_grad():
            sample = model.sample(sg, si, mk, num_steps=num_steps)       # (1, 4, 48, 48)
        generated[out_i] = sample.squeeze(0).cpu().numpy()

        if out_i % 50 == 0:
            elapsed = time.time() - t0
            eta     = elapsed / max(out_i, 1) * (N - out_i)
            print(f"  {out_i:4d}/{N}  ETA {eta:.0f}s")

    print(f"  Generation done in {time.time()-t0:.1f}s")
    return generated


# ---------------------------------------------------------------------------
# ResNet inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_resnet(model: nn.Module, images_np: np.ndarray,
               device: torch.device, batch_size: int = 256,
               already_normalized: bool = False) -> np.ndarray:
    """
    images_np: (N, 4, 48, 48) float32
    already_normalized: if True, skip (x - mean) / std (shard images are pre-normalized)
    Returns (N, 2) predictions [e1, e2]
    """
    model.eval()
    N   = len(images_np)
    out = np.zeros((N, 2), dtype=np.float32)
    for start in range(0, N, batch_size):
        end  = min(start + batch_size, N)
        imgs = images_np[start:end]                    # (B, 4, 48, 48)
        if not already_normalized:
            imgs = (imgs - NORM_MEAN) / NORM_STD
        t    = torch.from_numpy(imgs).to(device)
        pred = model(t)                                # (B, 2)
        out[start:end] = pred.cpu().numpy()
    return out


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def make_example_images(
    real_imgs: np.ndarray,
    gen_imgs: np.ndarray,
    true_e1: np.ndarray,
    true_e2: np.ndarray,
    pred_real: np.ndarray,
    pred_gen: np.ndarray,
    output_dir: Path,
    n_examples: int = 16,
    tag: str = "examples",
):
    """
    Grid of real vs generated images for visual sanity check.

    Each row is one galaxy:
      col 0: real target (RGB composite from I, R, G bands = idx 2, 1, 0)
      col 1: generated image (same composite)
      col 2: difference (generated - real), amplified

    Title of each row shows ground-truth and ResNet-predicted e1, e2.

    Visualization uses arcsinh stretch so faint features are visible.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    n = min(n_examples, len(real_imgs))

    def to_rgb(img_4ch):
        """4-channel (G,R,I,Z) → RGB using (I, R, G) = channels (2,1,0)."""
        rgb = img_4ch[[2, 1, 0], :, :]        # (3, H, W)
        rgb = np.arcsinh(rgb / 0.05) / 5.0    # arcsinh stretch
        rgb = np.clip((rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8), 0, 1)
        return np.transpose(rgb, (1, 2, 0))   # (H, W, 3)

    fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    col_titles = ["Real target", "Generated", "Difference (×5)"]
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=10, fontweight="bold")

    for row in range(n):
        real_rgb = to_rgb(real_imgs[row])
        gen_rgb  = to_rgb(gen_imgs[row])
        diff     = gen_imgs[row] - real_imgs[row]
        diff_rgb = np.transpose(diff[[2, 1, 0]], (1, 2, 0))
        # Centre diff around 0 for display
        d_scale  = max(abs(diff_rgb).max(), 1e-8)
        diff_disp = np.clip(diff_rgb / d_scale * 5.0 * 0.5 + 0.5, 0, 1)

        for col, img in enumerate([real_rgb, gen_rgb, diff_disp]):
            ax = axes[row, col]
            ax.imshow(img, interpolation="nearest")
            ax.axis("off")

        # Row label: true vs predicted e1, e2
        e1t, e2t = true_e1[row], true_e2[row]
        e1r, e2r = pred_real[row, 0], pred_real[row, 1]
        e1g, e2g = pred_gen[row, 0],  pred_gen[row, 1]
        label = (
            f"true: ({e1t:.3f}, {e2t:.3f})\n"
            f"real: ({e1r:.3f}, {e2r:.3f})\n"
            f"gen:  ({e1g:.3f}, {e2g:.3f})"
        )
        axes[row, 0].set_ylabel(label, fontsize=7, rotation=0,
                                 labelpad=90, va="center")

    fig.suptitle("Real vs Generated — visual sanity check\n"
                 "(e1, e2): true / ResNet on real / ResNet on generated",
                 fontsize=10)
    plt.tight_layout()
    out = output_dir / f"{tag}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def make_comparison_plots(
    true_e1, true_e2,
    pred_real_e1, pred_real_e2,
    pred_gen_e1,  pred_gen_e2,
    output_dir: Path,
    n_galaxies: int = 0,
    n_steps: int = 0,
):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)

    def r2(p, t):
        return 1.0 - np.sum((t - p)**2) / (np.sum((t - t.mean())**2) + 1e-12)

    # --- Overlaid scatter plots (real=green, generated=red, y=x black dashed) ---
    for comp_name, t, pr, pg in [
        ("e1", true_e1, pred_real_e1, pred_gen_e1),
        ("e2", true_e2, pred_real_e2, pred_gen_e2),
    ]:
        r2r = r2(pr, t)
        r2g = r2(pg, t)

        fig, ax = plt.subplots(figsize=(7, 7))
        lim = max(abs(t).max(), abs(pr).max(), abs(pg).max()) * 1.05

        ax.scatter(t, pr, s=8, alpha=0.5, color="#2ca02c", rasterized=True,
                   label=f"Real HSC images (R²={r2r:.4f})")
        ax.scatter(t, pg, s=8, alpha=0.5, color="#d62728", rasterized=True,
                   label=f"Generated HSC images (R²={r2g:.4f})")
        ax.plot([-lim, lim], [-lim, lim], "k--", lw=1.5, label="y = x")
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_xlabel(f"True SHAPE_{comp_name.upper()}", fontsize=13)
        ax.set_ylabel(f"Predicted SHAPE_{comp_name.upper()}", fontsize=13)
        ax.set_title(f"ResNet predictions — SHAPE_{comp_name.upper()}", fontsize=14)
        ax.legend(fontsize=11, loc="upper left")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3, lw=0.6)
        plt.tight_layout()
        out = output_dir / f"generated_vs_real_{comp_name}.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"  Saved: {out}  (real R²={r2r:.4f}, gen R²={r2g:.4f})")

    # --- Residual histograms + Gaussian fit ---
    for comp_name, t, pr, pg in [
        ("e1", true_e1, pred_real_e1, pred_gen_e1),
        ("e2", true_e2, pred_real_e2, pred_gen_e2),
    ]:
        res_real = pr - t
        res_gen  = pg - t
        fig, ax = plt.subplots(figsize=(8, 5))
        bins = np.linspace(-0.3, 0.3, 80)
        ax.hist(res_real, bins=bins, alpha=0.5, color="#2ca02c", density=True,
                label=f"Real HSC images ($\\sigma$={res_real.std():.4f})")
        ax.hist(res_gen,  bins=bins, alpha=0.5, color="#d62728", density=True,
                label=f"Generated HSC images ($\\sigma$={res_gen.std():.4f})")
        x = np.linspace(-0.3, 0.3, 300)
        ax.plot(x, stats.norm.pdf(x, loc=res_real.mean(), scale=res_real.std()),
                color="#2ca02c", lw=2, ls="--")
        ax.plot(x, stats.norm.pdf(x, loc=res_gen.mean(), scale=res_gen.std()),
                color="#d62728", lw=2, ls="--")
        ax.set_xlabel(f"Residual SHAPE_{comp_name.upper()} (predicted − true)", fontsize=12)
        ax.set_ylabel("Density", fontsize=12)
        ax.set_title(f"Residuals — SHAPE_{comp_name.upper()}", fontsize=13)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, lw=0.6)
        plt.tight_layout()
        out = output_dir / f"residuals_{comp_name}.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"  Saved: {out}")

    # --- Normality analysis: Q-Q plots + KS test ---
    N_QQ = min(5000, len(true_e1))
    rng = np.random.default_rng(42)
    theoretical = stats.norm.ppf(np.linspace(0.001, 0.999, N_QQ))

    for comp_name, t, pr, pg in [
        ("e1", true_e1, pred_real_e1, pred_gen_e1),
        ("e2", true_e2, pred_real_e2, pred_gen_e2),
    ]:
        res_real = pr - t
        res_gen  = pg - t

        # Standardize residuals for Q-Q
        z_real = (res_real - res_real.mean()) / (res_real.std() + 1e-12)
        z_gen  = (res_gen  - res_gen.mean())  / (res_gen.std()  + 1e-12)

        ks_real_stat, ks_real_p = stats.kstest(z_real, "norm")
        ks_gen_stat,  ks_gen_p  = stats.kstest(z_gen, "norm")

        fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

        # Left: overlaid histograms of standardized residuals vs N(0,1)
        ax = axes[0]
        bins = np.linspace(-5, 5, 100)
        ax.hist(z_real, bins=bins, density=True, alpha=0.5, color="#2ca02c",
                label=f"Real HSC images ($\\mu$={z_real.mean():.3f}, $\\sigma$={z_real.std():.3f})")
        ax.hist(z_gen, bins=bins, density=True, alpha=0.5, color="#d62728",
                label=f"Generated HSC images ($\\mu$={z_gen.mean():.3f}, $\\sigma$={z_gen.std():.3f})")
        x = np.linspace(-5, 5, 500)
        ax.plot(x, stats.norm.pdf(x), "k--", lw=1.8, label=r"$\mathcal{N}(0,\,1)$")
        ax.set_xlim(-5, 5)
        ax.set_xlabel("Standardized residual", fontsize=12)
        ax.set_ylabel("Density", fontsize=12)
        ax.set_title(f"Standardized residuals — SHAPE_{comp_name.upper()}", fontsize=13)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, lw=0.6)

        # Right: Q-Q plot
        ax = axes[1]
        n_qq = min(N_QQ, len(z_real))
        idx_real = rng.choice(len(z_real), n_qq, replace=False)
        idx_gen  = rng.choice(len(z_gen),  n_qq, replace=False)
        theo = theoretical[:n_qq]

        ax.scatter(theo, np.sort(z_real[idx_real]), s=4, alpha=0.5, color="#2ca02c",
                   label=f"Real HSC images (KS={ks_real_stat:.4f}, p={ks_real_p:.3g})")
        ax.scatter(theo, np.sort(z_gen[idx_gen]),   s=4, alpha=0.5, color="#d62728",
                   label=f"Generated HSC images (KS={ks_gen_stat:.4f}, p={ks_gen_p:.3g})")
        qlim = 4.5
        ax.plot([-qlim, qlim], [-qlim, qlim], "k--", lw=1.5, label="Perfect normality")
        ax.set_xlim(-qlim, qlim)
        ax.set_ylim(-qlim, qlim)
        ax.set_aspect("equal")
        ax.set_xlabel(r"Theoretical $\mathcal{N}(0,\,1)$ quantiles", fontsize=12)
        ax.set_ylabel("Empirical quantiles", fontsize=12)
        ax.set_title(f"Q-Q plot — SHAPE_{comp_name.upper()}", fontsize=13)
        ax.legend(fontsize=9, loc="upper left")
        ax.grid(True, alpha=0.3, lw=0.6)

        subtitle = f"{len(t)} galaxies, {n_steps} Euler steps"
        fig.text(0.5, -0.01, subtitle, ha="center", fontsize=10,
                 style="italic", color="gray")
        plt.tight_layout()
        out = output_dir / f"normality_{comp_name}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out}  (KS real={ks_real_stat:.4f} p={ks_real_p:.3g}, "
              f"KS gen={ks_gen_stat:.4f} p={ks_gen_p:.3g})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _step(name):
    """Print a timestamped step banner and return start time."""
    t = time.time()
    print(f"\n{'='*60}")
    print(f"[STEP] {name}")
    print(f"{'='*60}", flush=True)
    return t

def _done(name, t0):
    elapsed = time.time() - t0
    print(f"[DONE] {name} — {elapsed:.1f}s\n", flush=True)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    t_total = time.time()

    # --- Step 1: Load ResNet ---
    t0 = _step("Load trained ResNet")
    resnet = load_resnet(RESNET_CKPT, device)
    _done("Load trained ResNet", t0)

    # --- Step 2: Load shard ---
    t0 = _step("Load neighbors shard")
    targets, samegals, sameins, masks, surveys, meta_idx = load_shard(args.shard)
    _done("Load neighbors shard", t0)

    # --- Step 3: Filter HSC ---
    t0 = _step("Filter to HSC galaxies")
    hsc_mask = np.array([
        s.decode() if isinstance(s, bytes) else str(s)
        for s in surveys
    ]) == "hsc"
    hsc_indices = np.where(hsc_mask)[0]
    print(f"  HSC galaxies in shard: {len(hsc_indices)}/{len(targets)}")
    if args.n_galaxies is not None:
        hsc_indices = hsc_indices[:args.n_galaxies]
    print(f"  Using {len(hsc_indices)} HSC galaxies")
    _done("Filter to HSC galaxies", t0)

    # --- Step 4: Crossmatch IDs ---
    t0 = _step("Crossmatch IDs for ground-truth ellipticities")
    lookup      = build_id_to_ellipticity_lookup(RESNET_DATA)
    legacy_ids  = load_legacy_object_ids(NEIGHBOURS_H5)
    sub_meta_idx = meta_idx[hsc_indices]
    true_e1, true_e2, valid = resolve_ellipticities(sub_meta_idx, legacy_ids, lookup)
    hsc_indices = hsc_indices[valid]
    true_e1     = true_e1[valid]
    true_e2     = true_e2[valid]
    sub_meta_idx = sub_meta_idx[valid]
    print(f"  After crossmatch: {len(hsc_indices)} galaxies with known ellipticity")
    _done("Crossmatch IDs for ground-truth ellipticities", t0)

    # --- Step 5: ResNet on real targets ---
    t0 = _step("Run ResNet on real target images")
    real_imgs  = targets[hsc_indices]
    pred_real  = run_resnet(resnet, real_imgs, device, already_normalized=True)
    _done("Run ResNet on real target images", t0)

    # --- Step 6: Load flow-matching model ---
    t0 = _step("Load flow-matching model")
    flow_model = load_flow_model(args.checkpoint, device)
    _done("Load flow-matching model", t0)

    # --- Step 7: Generate images ---
    t0 = _step(f"Generate {len(hsc_indices)} images ({args.steps} Euler steps each)")
    generated  = generate_one_per_galaxy(
        flow_model, targets, samegals, sameins, masks, hsc_indices,
        num_steps=args.steps, device=device,
        compile_model=(not args.no_compile),
    )
    _done(f"Generate {len(hsc_indices)} images", t0)

    # --- Step 8: ResNet on generated images ---
    t0 = _step("Run ResNet on generated images")
    pred_gen = run_resnet(resnet, generated, device, already_normalized=True)
    _done("Run ResNet on generated images", t0)

    # --- Step 9: Summary stats ---
    t0 = _step("Compute summary statistics")
    for comp_name, t, pr, pg in [
        ("e1", true_e1, pred_real[:, 0], pred_gen[:, 0]),
        ("e2", true_e2, pred_real[:, 1], pred_gen[:, 1]),
    ]:
        r2r = 1.0 - np.sum((t - pr)**2) / (np.sum((t - t.mean())**2) + 1e-12)
        r2g = 1.0 - np.sum((t - pg)**2) / (np.sum((t - t.mean())**2) + 1e-12)
        bias_real = np.mean(pr - t)
        bias_gen  = np.mean(pg - t)
        print(f"  SHAPE_{comp_name.upper()}:  real R²={r2r:.4f} bias={bias_real:+.4f} | "
              f"gen R²={r2g:.4f} bias={bias_gen:+.4f}")
    _done("Compute summary statistics", t0)

    # --- Step 10: Plots ---
    out_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = _step(f"Save plots to {out_dir}")
    make_example_images(
        real_imgs, generated,
        true_e1, true_e2,
        pred_real, pred_gen,
        out_dir,
        n_examples=16,
        tag="example_images",
    )
    make_comparison_plots(
        true_e1, true_e2,
        pred_real[:, 0], pred_real[:, 1],
        pred_gen[:, 0],  pred_gen[:, 1],
        out_dir,
        n_galaxies=len(hsc_indices),
        n_steps=args.steps,
    )
    _done("Save plots", t0)

    total = time.time() - t_total
    print(f"\n{'='*60}")
    print(f"ALL DONE — total wall time: {total:.1f}s")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate ResNet ellipticity on flow-matching generated images"
    )
    parser.add_argument(
        "--shard", required=True,
        help="Path to a neighbors shard HDF5 (e.g. neighbors_shard_0000.h5)",
    )
    parser.add_argument(
        "--checkpoint", default=DEFAULT_CHECKPOINT,
        help="Path to flow-matching Lightning checkpoint",
    )
    parser.add_argument(
        "--n-galaxies", type=int, default=None,
        help="Max HSC galaxies to process (default: all HSC in shard)",
    )
    parser.add_argument(
        "--steps", type=int, default=250,
        help="Euler integration steps for generation (default: 250)",
    )
    parser.add_argument(
        "--no-compile", action="store_true",
        help="Disable torch.compile on velocity_model",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Output directory for plots (default: resnet_experiment/outputs/)",
    )
    args = parser.parse_args()
    main(args)
