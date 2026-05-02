"""
Stage 4 — STUB. Implement on engaging.

Read all per-checkpoint prediction CSVs and produce two plot artifacts:

1. Per-checkpoint plot (compare_<ckpt>.png), one per checkpoint:
     Horizontal-bar layout, modeled on ../final/makeplot_v2.py:
       - rows grouped by task family with band-averaged target labels
         (use the same regex as makeplot_v2.py to collapse e.g. a_g/r/i/z/y → "extinction")
       - each row has up to 3 bars hued by latent_variant
         (HSC = blue, Legacy = orange, Combined = green)
       - shaded background bands per task family (physics, instrument, morphology)

2. Cross-checkpoint plot (compare_all_checkpoints.png):
     Same layout but each (task, target, latent) gets one bar per checkpoint
     (color = checkpoint, linestyle/hatch = latent_variant), so all four
     variants in the registry can be compared at a glance.

CLI shape (locked):

    python makeplot.py \
        --predictions-dir outputs/predictions \
        --out-dir         outputs/plots \
        [--checkpoints base hier hier-small single-baseline]

Reuses:
- ../final/makeplot_v2.py for color palette, group-bg shading, band-averaging
  regex, and figure size heuristics. Copy and adapt; don't import — the file
  uses module-level state.
"""

import argparse
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--predictions-dir", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--checkpoints", nargs="+",
                   default=["base", "hier", "hier-small", "single-baseline"])
    args = p.parse_args()

    raise NotImplementedError(
        "TODO(engaging): implement plotting. See module docstring for the contract.\n"
        f"Args: {vars(args)}"
    )


if __name__ == "__main__":
    main()
