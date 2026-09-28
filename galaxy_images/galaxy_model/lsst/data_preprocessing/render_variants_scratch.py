"""Display-only rendering variants for the 48x48 griz tensors (HSC + LSST).

Nothing here touches training: every variant is a function from the stored
tensor (what the model sees) to an RGB picture (what a human sees).
Rows = variants, columns = 4 COSMOS galaxies x (HSC, LSST, nearest LSST neighbour).
"""
import json, os, sys, textwrap
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.visualization import make_lupton_rgb

D = "/work1/jeroenaudenaert/pablomer/data"
OUT = sys.argv[1] if len(sys.argv) > 1 else "gallery_variants.png"

# ------------------------------------------------------------- load tensors
nb = np.load(f"{D}/lsst_efficient/lsst_neighbors.npz"); idx, pr = nb["idx"], nb["parent_row"]
meta = pd.read_parquet(f"{D}/lsst_efficient/lsst_meta.parquet",
                       columns=["object_id_hsc", "field", "n_input_g", "n_input_r", "n_input_i", "n_input_z"])
ok = (meta.field == "COSMOS").to_numpy()
for b in "griz":
    ok = ok & (meta[f"n_input_{b}"].to_numpy() >= 5)
L = np.memmap(f"{D}/lsst_efficient/lsst48.bin", dtype="float16", mode="r", shape=(65163, 4, 48, 48))
H = np.memmap(f"{D}/efficient_neighs_48/hsc48.bin", dtype="float16", mode="r", shape=(468197, 4, 48, 48))

rng = np.random.default_rng(1)                     # same seed as the existing gallery
anchors = rng.choice(np.where(ok)[0], 6, replace=False)[:4]
panels = []                                        # (label, survey, tensor)
for a in anchors:
    nb1 = next(j for j in idx[a] if ok[j])
    panels += [("HSC", "hsc", np.asarray(H[pr[a]], np.float32)),
               ("LSST", "lsst", np.asarray(L[a], np.float32)),
               ("LSST nb", "lsst", np.asarray(L[nb1], np.float32))]

# ------------------------------------------------ tensor -> flux inversion
NORM = {"hsc": (0.022, 0.05), "lsst": (0.0303, 0.0591)}
def to_flux(x, survey):
    """Undo z-score and arcsinh: back to (clamped, resampled) flux, nanomaggy-like."""
    m, s = NORM[survey]
    y = x * s + m                                  # arcsinh domain: arcsinh(f/0.01)*0.1
    return 0.01 * np.sinh(y / 0.1)
def irg(x):                                        # (4,H,W) griz -> (i, r, g)
    return x[2], x[1], x[0]

# global display window for the "fixed window" variant, from a 400-stamp sample
samp = rng.choice(np.where(ok)[0], 400, replace=False)
allv = np.concatenate([np.asarray(L[samp], np.float32).ravel(), np.asarray(H[pr[samp]], np.float32).ravel()])
G_LO, G_HI = np.percentile(allv, 0.5), np.percentile(allv, 99.9)

# ----------------------------------------------------------------- variants
def v_current(x, survey):
    img = np.stack(irg(x), -1); lo, hi = np.percentile(img, 1), np.percentile(img, 99.5)
    return np.clip((img - lo) / max(hi - lo, 1e-6), 0, 1)

def v_fixed(x, survey, gamma=0.7):
    img = np.stack(irg(x), -1)
    return np.clip((img - G_LO) / (G_HI - G_LO), 0, 1) ** gamma

def v_lupton(stretch, Q, minimum):
    def f(x, survey):
        i, r, g = irg(to_flux(x, survey))
        return make_lupton_rgb(i, r, g, minimum=minimum, stretch=stretch, Q=Q) / 255.0
    return f

def v_lupton_auto(x, survey, Q=8):
    fl = to_flux(x, survey); i, r, g = irg(fl)
    s = max(np.percentile(fl, 99.5), 1e-3)         # per-panel stretch: every galaxy fills the range
    return make_lupton_rgb(i, r, g, minimum=0.0, stretch=s, Q=Q) / 255.0

def v_gray_i(x, survey):
    fl = to_flux(x, survey)[2]
    y = np.arcsinh(np.clip(fl, -0.002, None) / 0.004)  # asinh with a fixed soft knee
    y = (y - np.arcsinh(-0.5)) / (np.arcsinh(0.25 / 0.004) - np.arcsinh(-0.5))
    return plt.get_cmap("gray")(np.clip(y, 0, 1))[..., :3]

VARIANTS = [
    ("A  current", "per-stamp 1–99.5 % stretch of the model tensor; nearest pixels", v_current, "nearest"),
    ("B  current, smoothed", "identical mapping to A, only the up-sampling filter changed", v_current, "lanczos"),
    ("C  fixed window", "model tensor, ONE display window for all stamps + both surveys (γ=0.7): brightness and noise comparable across panels", v_fixed, "lanczos"),
    ("D  Lupton, soft", "flux-space asinh RGB (Lupton+04), shared stretch=0.05 Q=8, min=0: noise stays dark, colours preserved", v_lupton(0.05, 8, 0.0), "lanczos"),
    ("E  Lupton, crisp", "as D with stretch=0.02 Q=10 and min=+0.5σ: faint outskirts up, sky clipped to black", v_lupton(0.02, 10, 0.0005), "lanczos"),
    ("F  Lupton, auto", "per-stamp stretch (99.5 % flux): every galaxy fills the range; hides brightness differences", v_lupton_auto, "lanczos"),
    ("G  i-band gray", "single band, fixed asinh knee: morphology only, no colour noise", v_gray_i, "lanczos"),
]

# -------------------------------------------------------------------- figure
ncol = len(panels); nrow = len(VARIANTS)
fig = plt.figure(figsize=(1.18 * ncol + 3.2, 1.22 * nrow + 0.9))
gs = fig.add_gridspec(nrow, ncol + 1, width_ratios=[2.9] + [1] * ncol, wspace=0.04, hspace=0.06,
                      left=0.01, right=0.995, top=0.94, bottom=0.01)
for r, (name, desc, fn, interp) in enumerate(VARIANTS):
    ax = fig.add_subplot(gs[r, 0]); ax.axis("off")
    ax.text(0.0, 0.62, name, fontsize=10.5, fontweight="bold", va="center", transform=ax.transAxes)
    ax.text(0.0, 0.34, textwrap.fill(desc, 46), fontsize=6.6, va="center", linespacing=1.25, transform=ax.transAxes)
    for c, (lab, survey, x) in enumerate(panels):
        ax = fig.add_subplot(gs[r, c + 1]); ax.set_xticks([]); ax.set_yticks([])
        ax.imshow(fn(x, survey), origin="lower", interpolation=interp)
        if r == 0:
            ax.set_title(lab, fontsize=8, pad=2)
        if c % 3 == 0:
            for sp in ax.spines.values(): sp.set_edgecolor("#2a78d6"); sp.set_linewidth(1.4)
fig.suptitle("Rendering variants for the same 48×48 griz tensors — 4 COSMOS galaxies, each as HSC | LSST | nearest LSST neighbour. "
             "Display only; none of this changes what the model receives.", fontsize=9)
fig.savefig(OUT, dpi=150)
print("wrote", OUT, f"| global window for C: [{G_LO:.2f}, {G_HI:.2f}] normalised units")
