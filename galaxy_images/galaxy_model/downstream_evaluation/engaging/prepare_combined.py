"""
Stage 2 — STUB. Implement on engaging.

Embed the three subsets (overlap / hsc_only / legacy_only) with one checkpoint
and write a single H5 with the layout below.

CLI shape (locked — predict_combined.py and the slurm wrappers depend on it):

    python prepare_combined.py \
        --checkpoint   <path-to-snapshot.ckpt> \
        --module       <module-relative-to-galaxy_model/> \
        --model-class  <ClassName> \
        --index-dir    outputs/index \
        --out          outputs/embeddings/<ckpt_name>.h5 \
        [--hsc-dir    /work1/.../hsc_downstream] \
        [--legacy-dir /work1/.../legacy_downstream_quick] \
        [--batch-size 256] \
        [--smoke]              # if set, take first 512 rows of each subset only

Per-checkpoint H5 layout:

    /overlap/
        hsc_e1, hsc_e2                        (N, D1), (N, D2)   encoder_1/2 on HSC image
        legacy_e1, legacy_e2                  (N, D1), (N, D2)   encoder_1/2 on Legacy image
        combined_e1, combined_e2              (N, 2*D1), (N, 2*D2)   concat([hsc_e*, legacy_e*])
        labels/<col>                          per HSC catalog + per Legacy catalog,
                                              prefixed "hsc_" / "legacy_" to disambiguate
    /hsc_only/
        hsc_e1, hsc_e2
        labels/<col>                          HSC catalog only (prefix "hsc_")
    /legacy_only/
        legacy_e1, legacy_e2
        labels/<col>                          Legacy catalog only (prefix "legacy_")

For the single-encoder variant (--model-class SingleEncoderFlowMatchingModule),
write only *_e1 (skip *_e2).

Reuses (copy-paste from these — they're already correct):
- HSCBinaryDataset, _detect_n_galaxies, load_trained_model, load_untrained_model:
    ../new_data_downstream/prepare_hsc_downstream.py
- preprocess_image_v2(img, crop_size=48, survey="hsc"|"legacy"):
    galaxy_images/image_preprocessing.py
- ROCm hipblaslt workaround at top of file (only matters on AMD nodes; harmless on H100).

Implementation outline:
1. Resolve paths, set device, load index parquets (overlap.parquet, hsc_only.parquet,
   legacy_only.parquet) from --index-dir. Apply --smoke truncation if requested.
2. For each subset, build a torch Dataset over the appropriate .bin file(s):
     - overlap   needs both HSC and Legacy reads keyed by (hsc_image_idx, legacy_image_idx)
     - hsc_only  needs HSC reads only (key: hsc_image_idx)
     - legacy_only needs Legacy reads only (key: legacy_image_idx)
   Generalize HSCBinaryDataset into a per-survey BinaryDataset that takes a
   (bin_path, indices, bands, h, w) — pre-existing constants for HSC are in
   prepare_hsc_downstream.py; Legacy is (4, 160, 160) float16, no y-band drop,
   normalisation TBD on engaging (start with the same arcsinh+normalize and
   per-survey mean/std from prepare_hsc_downstream.py — for Legacy, see what
   ../predict_legacy_provabgs.py does).
3. Load model once; for each subset, run encoder_1 / encoder_2 via the
   pattern at prepare_hsc_downstream.py:194-210. For overlap, run twice
   (HSC images then Legacy images on the same model) and concat per-encoder
   to form combined_e*.
4. Pull labels from both catalogs by joining on image_idx, prefix the column
   names with "hsc_" / "legacy_", use the SKIP_COLS pattern from
   prepare_hsc_downstream.py (extend with Legacy-side IDs: BRICKID, BRICKNAME,
   OBJID, RELEASE, gid, etc.).
5. Save with the layout above. Always set h5 attrs:
       embedding_names = list of encoder names actually written
       num_examples_per_subset = {"overlap": ..., "hsc_only": ..., "legacy_only": ...}
       label_columns_per_subset = {...}
       checkpoint_path = args.checkpoint
       checkpoint_name = Path(args.checkpoint).parts[-2]   # registry variant name

Sanity check before declaring done:
- For the smoke run on `base`, hsc_e2 instrument-task R² (from predict_combined.py)
  should match within ~5% the existing ../new_data_downstream/predict_hsc_downstream
  numbers on the same checkpoint and the same HSC galaxies. If it doesn't, the
  preprocessing path is off.
"""

import argparse
import sys
from pathlib import Path

# ROCm workaround (harmless on H100).
try:
    import torch  # noqa: F401
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "preferred_blas_library"):
        torch.backends.cuda.preferred_blas_library("hipblas")
except Exception:
    pass

_here = Path(__file__).resolve().parent
_project_root = _here.parents[3]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--module", required=True,
                   help="Path relative to galaxy_model/ (e.g. 'double_train_fm_neighbors.py' or "
                        "'hierarchical_attention/double_train_fm_neighbors_hier_global_ins.py')")
    p.add_argument("--model-class", required=True,
                   help="LightningModule class name inside --module")
    p.add_argument("--index-dir", required=True, type=Path,
                   help="Directory containing overlap.parquet, hsc_only.parquet, legacy_only.parquet")
    p.add_argument("--out", required=True, type=Path,
                   help="Output H5 path, e.g. outputs/embeddings/base.h5")
    p.add_argument("--hsc-dir", type=Path,
                   default=Path("/work1/jeroenaudenaert/pablomer/data/hsc_downstream"))
    p.add_argument("--legacy-dir", type=Path,
                   default=Path("/work1/jeroenaudenaert/pablomer/data/legacy_downstream_quick"))
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--smoke", action="store_true",
                   help="Use only the first 512 rows of each subset")
    args = p.parse_args()

    raise NotImplementedError(
        "TODO(engaging): implement the embedding loop. See module docstring for the contract.\n"
        f"Args: {vars(args)}"
    )


if __name__ == "__main__":
    main()
