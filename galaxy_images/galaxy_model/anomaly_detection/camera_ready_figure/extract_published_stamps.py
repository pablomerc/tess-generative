"""Pull the 36 top-anomaly stamps out of the published Fig. 3 PDF.

The published figure's score files and `_cache/` live on AMD only, so instead of
re-deriving rows 1-4 we reuse the exact RGB stamps embedded in the paper PDF
(matplotlib embedded each imshow as a 251x251 RGB image; the "#i" labels are
separate vector text, so the images are clean).

Needs PyMuPDF, which is in the pptx venv (not torchenv):
    ~/venvs/pptx/bin/python extract_published_stamps.py

Writes `published_top12_stamps.npz`:
    rgb    uint8 (3 groups, 12 ranks, 251, 251, 3), group order Physics | AION-1 | Instrument
    bbox   float (3, 12, 4) page bbox of each stamp, for provenance
"""
from pathlib import Path

import numpy as np
import pymupdf

HERE = Path(__file__).resolve().parent
PAPER_PDF = Path("/orcd/pool/007/pablomer/tess-gen-disentanglement_neurips2026_paper/"
                 "figures/new_figures/anomaly_detection_figure_science.pdf")
N_GROUPS, N_ROWS, N_COLS_PER = 3, 4, 3


def main():
    doc = pymupdf.open(PAPER_PDF)
    info = doc[0].get_image_info(xrefs=True)
    assert len(info) == N_GROUPS * N_ROWS * N_COLS_PER, len(info)

    # Grid position from the page bbox: 9 distinct x0 (columns), 4 distinct y0 (rows).
    x0s = np.unique(np.round([d["bbox"][0] for d in info], 1))
    y0s = np.unique(np.round([d["bbox"][1] for d in info], 1))
    assert len(x0s) == N_GROUPS * N_COLS_PER and len(y0s) == N_ROWS, (x0s, y0s)

    rgb = np.zeros((N_GROUPS, N_ROWS * N_COLS_PER, 251, 251, 3), dtype=np.uint8)
    bbox = np.zeros((N_GROUPS, N_ROWS * N_COLS_PER, 4))
    seen = set()
    for d in info:
        col = int(np.searchsorted(x0s, round(d["bbox"][0], 1)))
        row = int(np.searchsorted(y0s, round(d["bbox"][1], 1)))  # PyMuPDF y grows downwards
        g, k = col // N_COLS_PER, row * N_COLS_PER + col % N_COLS_PER
        pix = pymupdf.Pixmap(doc, d["xref"])
        assert (pix.n, pix.alpha, pix.width, pix.height) == (3, 0, 251, 251)
        rgb[g, k] = np.frombuffer(pix.samples, dtype=np.uint8).reshape(251, 251, 3)
        bbox[g, k] = d["bbox"]
        seen.add((g, k))
    assert len(seen) == rgb.shape[0] * rgb.shape[1]

    out = HERE / "published_top12_stamps.npz"
    np.savez_compressed(out, rgb=rgb, bbox=bbox, source_pdf=str(PAPER_PDF))
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
