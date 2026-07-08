"""
make_figure_no_legacy.py — variant of make_figure.py without the Legacy-query row.

Produces a figure with only the HSC-query row (Panel A) and the Lens row (Panel B).

Usage:
    cd figures_for_paper/nearest_neighbors_figure
    python make_figure_no_legacy.py
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec

_LIBERATION_DIR = Path("/usr/share/fonts/liberation-serif")
for _fname in ("LiberationSerif-Regular.ttf", "LiberationSerif-Bold.ttf",
               "LiberationSerif-Italic.ttf", "LiberationSerif-BoldItalic.ttf"):
    _p = _LIBERATION_DIR / _fname
    if _p.exists():
        fm.fontManager.addfont(str(_p))

_here     = Path(__file__).resolve().parent
CACHE_DIR = _here / "_cache"

LENS_RANKS = [4, 5, 6, 7, 9, 11, 12]


def lens_display_rank(r):
    return r - 3 if r < 9 else r - 4


def row_scale_rgb(arr_chw, vmin, vmax):
    x = arr_chw[:3].copy()
    v0 = np.array(vmin).reshape(3, 1, 1)
    v1 = np.array(vmax).reshape(3, 1, 1)
    return np.clip((x - v0) / (v1 - v0 + 1e-8), 0.0, 1.0).transpose(1, 2, 0)


def percentile_rgb(arr_chw, lo=0.5, hi=99.5):
    rgb = arr_chw[:3].copy().transpose(1, 2, 0)
    for i in range(3):
        p_lo = np.percentile(rgb[:, :, i], lo)
        p_hi = np.percentile(rgb[:, :, i], hi)
        rgb[:, :, i] = np.clip(rgb[:, :, i], p_lo, p_hi)
        ch = rgb[:, :, i]; lo2, hi2 = ch.min(), ch.max()
        rgb[:, :, i] = (ch - lo2) / (hi2 - lo2) if hi2 > lo2 else 0.0
    return rgb


def show(ax, rgb_hwc, label=None, label_color="black", fontsize=12):
    ax.imshow(rgb_hwc, interpolation="nearest")
    if label:
        ax.text(0.5, 0.96, label, transform=ax.transAxes,
                fontsize=fontsize, fontweight="bold", color=label_color,
                va="top", ha="center",
                bbox=dict(boxstyle="square,pad=0.2", facecolor="white",
                          alpha=0.65, linewidth=0))
    ax.set_axis_off()


def fmt(src): return "(HSC)" if src == "hsc" else "(Leg)"


def load_query172():
    d = np.load(CACHE_DIR / "query_172.npz", allow_pickle=False)
    return {k: d[k] for k in d.files}


def load_lens48():
    d = np.load(CACHE_DIR / "lens_48.npz", allow_pickle=False)
    return {
        "obj_id":    str(d["obj_id"]),
        "query_img": d["query_img"],
        "nn_survey": [str(s) for s in d["nn_survey"]],
        "nn_imgs":   d["nn_imgs"],
    }


def make_figure(out_png, out_pdf, fontsize_offset=0):
    q    = load_query172()
    lens = load_lens48()

    q_hsc = q["query_hsc"]
    vmin  = q_hsc[:3].reshape(3, -1).min(axis=1)
    vmax  = q_hsc[:3].reshape(3, -1).max(axis=1)
    def nn_rgb(arr): return row_scale_rgb(arr, vmin, vmax)

    n_phys   = len(q["hsc_phys_ranks"])    # 5
    n_inst   = len(q["hsc_inst_ranks"])    # 2
    n_cols_A = 1 + n_phys + n_inst         # 8
    n_cols_B = 1 + len(LENS_RANKS)         # 8
    n_cols   = n_cols_A

    img_w, img_h = 2.2, 2.2
    fig = plt.figure(figsize=(n_cols * img_w, 2 * img_h + 0.9))

    # Rows: title | HSC | gap | Lens
    gs = GridSpec(4, n_cols, figure=fig,
                  height_ratios=[0.18, 1, 0.20, 1],
                  hspace=0.06, wspace=0.06,
                  top=0.97, bottom=0.02, left=0.01, right=0.99)

    axes_hsc  = [fig.add_subplot(gs[1, c]) for c in range(n_cols)]
    axes_lens = [fig.add_subplot(gs[3, c]) for c in range(n_cols_B)]
    for c in range(n_cols):
        fig.add_subplot(gs[2, c]).set_axis_off()
    for c in range(n_cols_B, n_cols):
        fig.add_subplot(gs[3, c]).set_axis_off()

    fs_label = 12 + fontsize_offset
    fs_title = 14 + fontsize_offset
    fs_lens_query = 8 + fontsize_offset

    # Panel A — HSC row only
    show(axes_hsc[0], nn_rgb(q["query_hsc"]), "HSC", fontsize=fs_label)

    for i, (img, src, rank) in enumerate(zip(
            q["hsc_phys_imgs"], q["hsc_phys_srcs"], q["hsc_phys_ranks"])):
        color = "indianred" if rank == 1 else ("#B8860B" if src == "legacy" else "black")
        show(axes_hsc[1+i], nn_rgb(img), f"NN#{rank} {fmt(src)}", color, fontsize=fs_label)

    for i, (img, src, rank) in enumerate(zip(
            q["hsc_inst_imgs"], q["hsc_inst_srcs"], q["hsc_inst_ranks"])):
        color = "#B8860B" if src == "legacy" else "black"
        show(axes_hsc[1+n_phys+i], nn_rgb(img), f"NN#{rank} {fmt(src)}", color, fontsize=fs_label)

    # Panel B
    show(axes_lens[0], percentile_rgb(lens["query_img"]),
         f"HSC {lens['obj_id']}", fontsize=fs_lens_query)
    for c, r in enumerate(LENS_RANKS, start=1):
        src   = lens["nn_survey"][r - 1]
        img   = lens["nn_imgs"][r - 1]
        color = "#B8860B" if src == "legacy" else "black"
        show(axes_lens[c], percentile_rgb(img), f"NN #{lens_display_rank(r)} {fmt(src)}", color,
             fontsize=fs_label)

    # Titles
    fig.canvas.draw()
    def col_cx(ax): return (ax.get_position().x0 + ax.get_position().x1) / 2

    title_y = axes_hsc[0].get_position().y1 + 0.012
    fig.text(col_cx(axes_hsc[0]), title_y, "Query",
             ha="center", va="bottom", fontsize=fs_title, fontweight="bold")
    fig.text((axes_hsc[1].get_position().x0 + axes_hsc[n_phys].get_position().x1) / 2,
             title_y, "Physics NNs",
             ha="center", va="bottom", fontsize=fs_title, fontweight="bold", color="#2E86AB")
    fig.text((axes_hsc[1+n_phys].get_position().x0 + axes_hsc[-1].get_position().x1) / 2,
             title_y, "Instrument NNs",
             ha="center", va="bottom", fontsize=fs_title, fontweight="bold", color="#CC546D")

    lens_title_y = axes_lens[0].get_position().y1 + 0.012
    fig.text(col_cx(axes_lens[0]), lens_title_y, "Query",
             ha="center", va="bottom", fontsize=fs_title, fontweight="bold")
    fig.text((axes_lens[1].get_position().x0 + axes_lens[-1].get_position().x1) / 2,
             lens_title_y, "Physics NNs, HSC only",
             ha="center", va="bottom", fontsize=fs_title, fontweight="bold", color="#2E86AB")

    # Dashed separators — Panel A (vertical lines now span the single HSC row)
    y_top = axes_hsc[0].get_position().y1 + 0.005
    y_bot = axes_hsc[0].get_position().y0 - 0.005

    def vline(x0, x1, y0, y1):
        xm = (x0 + x1) / 2
        fig.add_artist(Line2D([xm, xm], [y0, y1], transform=fig.transFigure,
                              color="black", linewidth=2.5, linestyle="--"))

    vline(axes_hsc[0].get_position().x1,      axes_hsc[1].get_position().x0,      y_bot, y_top)
    vline(axes_hsc[n_phys].get_position().x1, axes_hsc[1+n_phys].get_position().x0, y_bot, y_top)

    # Dashed separator — Panel B
    y_top_b = axes_lens[0].get_position().y1 + 0.005
    y_bot_b = axes_lens[0].get_position().y0 - 0.005
    vline(axes_lens[0].get_position().x1, axes_lens[1].get_position().x0, y_bot_b, y_top_b)

    # Horizontal rule between panels
    gap_y0 = axes_lens[0].get_position().y1
    gap_y1 = axes_hsc[0].get_position().y0
    rule_y = gap_y1 - 0.25 * (gap_y1 - gap_y0)
    fig.add_artist(Line2D([0.01, 0.99], [rule_y, rule_y], transform=fig.transFigure,
                          color="#aaaaaa", linewidth=2.5, linestyle="-"))

    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_png}")
    print(f"Saved: {out_pdf}")


if __name__ == "__main__":
    base = _here / "nearest_neighbors_figure_no_legacy"

    # Liberation Serif version
    matplotlib.rcParams["font.family"]      = "serif"
    matplotlib.rcParams["font.serif"]       = ["Liberation Serif", "DejaVu Serif"]
    matplotlib.rcParams["mathtext.fontset"] = "stix"
    make_figure(base.with_suffix(".png"), base.with_suffix(".pdf"), fontsize_offset=5)

    # scienceplots version
    import scienceplots  # noqa: F401
    with plt.style.context(["science", "no-latex"]):
        make_figure(
            base.parent / (base.name + "_science.png"),
            base.parent / (base.name + "_science.pdf"),
            fontsize_offset=5,
        )
