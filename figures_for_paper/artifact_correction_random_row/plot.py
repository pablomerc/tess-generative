"""Plot a single-row version of the exp_E artifact-correction figure.

Loads the saved tensors from the exp_E *repeat* run (10 instrument anomalies x
10 random HSC pairs, with single + mean-of-5 reconstructions) and renders a
1xN grid showing the targets followed by the mean reconstructions.

Source tensors keys (exp_E_repeat_random_10pairs/tensors.npz):
    instrument_ranks (10,)         AION instrument-anomaly ranks
    rand_raw_idxs    (10,)         random HSC raw indices used as sameins pool
    ins_hsc          (10,4,48,48)  target HSC images (one per rank)
    rand_hsc         (10,4,48,48)  the random HSC pool images
    recons_single    (10,4,48,48)  single-noise reconstructions
    recons_mean      (10,4,48,48)  mean over 5 posterior samples (used here)
    recons_all       (10,5,4,48,48)
"""
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent

# Rows in the source 10-row figure, 1-indexed top-to-bottom.
SELECTED_ROWS_1INDEXED = [2, 3, 5, 6]


def to_rgb(img_chw: np.ndarray) -> np.ndarray:
    """Percentile (1-99) stretch on first 3 channels -> HxWx3 in [0, 1]."""
    rgb = img_chw[:3].astype(np.float32)
    lo = np.percentile(rgb, 1, axis=(1, 2), keepdims=True)
    hi = np.percentile(rgb, 99, axis=(1, 2), keepdims=True)
    return np.clip((rgb - lo) / (hi - lo + 1e-8), 0, 1).transpose(1, 2, 0)


def main() -> None:
    data = np.load(HERE / "tensors.npz")
    ins_hsc = data["ins_hsc"]
    recons_mean = data["recons_mean"]
    instrument_ranks = data["instrument_ranks"]

    idxs = [r - 1 for r in SELECTED_ROWS_1INDEXED]
    targets = [ins_hsc[i] for i in idxs]
    reconstructions = [recons_mean[i] for i in idxs]
    ranks = [int(instrument_ranks[i]) for i in idxs]

    n = len(idxs)
    img_size = 1.6
    pair_gap = 0.10   # between pair-groups, in image-size units
    inner_gap = 0.04  # inside a pair (original next to corrected)

    fig_w = n * 2 * img_size + (n - 1) * pair_gap * img_size
    fig_h = img_size + 0.6
    fig = plt.figure(figsize=(fig_w, fig_h))

    outer = gridspec.GridSpec(
        1, n, figure=fig,
        left=0.005, right=0.995, top=0.78, bottom=0.02,
        wspace=pair_gap,
    )

    for j, (orig, corr) in enumerate(zip(targets, reconstructions)):
        inner = gridspec.GridSpecFromSubplotSpec(
            1, 2, subplot_spec=outer[j], wspace=inner_gap,
        )
        ax_o = fig.add_subplot(inner[0, 0])
        ax_o.imshow(to_rgb(orig))
        ax_o.set_title(f"Original image {j + 1}", fontsize=8, pad=2)
        ax_o.axis("off")

        ax_c = fig.add_subplot(inner[0, 1])
        ax_c.imshow(to_rgb(corr))
        ax_c.set_title(f"Corrected image {j + 1}", fontsize=8, pad=2)
        ax_c.axis("off")

    # Thin vertical separators between adjacent pairs (figure coords).
    fig.canvas.draw()
    for j in range(n - 1):
        right_pos = outer[j].get_position(fig)
        left_pos = outer[j + 1].get_position(fig)
        x_mid = (right_pos.x1 + left_pos.x0) / 2
        y0 = right_pos.y0
        y1 = right_pos.y1
        line = plt.Line2D(
            [x_mid, x_mid], [y0, y1],
            transform=fig.transFigure,
            color="0.7", linewidth=0.6, zorder=10,
        )
        fig.add_artist(line)
    out_png = HERE / "artifact_correction_random_row.png"
    out_pdf = HERE / "artifact_correction_random_row.pdf"
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_png}")
    print(f"Saved: {out_pdf}")


if __name__ == "__main__":
    main()
