"""
Stage 3 — STUB. Implement on engaging.

Read one checkpoint's H5 (from prepare_combined.py) and train MLP probes
across (task_family, target, latent_variant). Write a long-form CSV.

CLI shape (locked):

    python predict_combined.py \
        --embeddings outputs/embeddings/<ckpt_name>.h5 \
        --out        outputs/predictions/predict_<ckpt_name>.csv \
        [--device cuda] [--max-epochs 50] [--seed 0]

CSV columns:
    checkpoint, task_family, target, latent_variant, score, score_metric,
    n_train, n_test

Tasks (allowed latent variants per task family — restrict to whichever
embeddings actually exist in the H5; single-baseline only has *_e1):

    | task_family         | subset(s)                      | latent variants                                       | metric            |
    |---------------------|--------------------------------|-------------------------------------------------------|-------------------|
    | physics_provabgs    | overlap                        | hsc_e1, legacy_e1, combined_e1                        | R² (regression)   |
    | morphology_gz10     | overlap ∪ hsc_only ∪ legacy_only (wherever label exists) | hsc_e1, legacy_e1, combined_e1 (overlap only) | accuracy + F1     |
    | instrument_hsc      | overlap.hsc + hsc_only         | hsc_e2                                                | R² (regression)   |
    | instrument_legacy   | overlap.legacy + legacy_only   | legacy_e2                                             | R² (regression)   |

Targets:
- physics_provabgs: desi_z, provabgs_logmstar, provabgs_tage_mw,
  provabgs_avg_sfr, provabgs_z_mw  (and ssfr derived: avg_sfr / 10**logmstar)
- morphology_gz10: gz10_label (10-class, classification)
- instrument_hsc: a_g, a_r, a_i, a_z, a_y,
  {g,r,i,z}_variance_value, psf_fwhm_{g,r,i,z}
- instrument_legacy: PSFSIZE_{G,R,I,Z}, PSFDEPTH_{G,R,I,Z},
  GALDEPTH_{G,R,I,Z}, MW_TRANSMISSION_{G,R,I,Z} (and other Legacy
  catalog instrument columns — pick on engaging from
  legacy_downstream_quick/catalog.parquet)

Reuses:
- ../new_data_downstream/predict_hsc_downstream.py for the MLP architecture,
  train/val split, EarlyStopping setup, and the load_h5_variants pattern.
  Generalize from a single embedding key to (subset, latent_variant) keys.

Implementation outline:
1. Open the H5 once, read attrs (label_columns_per_subset, embedding_names,
   checkpoint_name).
2. For each (task_family, target, latent_variant) tuple where both the latent
   AND the label exist (and are non-null on enough rows):
     a. Assemble (X, y) by filtering rows where target is finite.
     b. Train a small MLP (256, 128) with AdamW + EarlyStopping (patience ~5).
     c. Evaluate on a held-out 10% split (TRAIN_FRAC=0.9) and record metric.
3. Append rows to the long-form CSV. One CSV per checkpoint.

Notes:
- For combined_e* the input dim is 2x the per-survey dim — the MLP shape is
  the same; just sklearn will see a wider input.
- For classification tasks, use accuracy and macro-F1 columns; record both as
  separate rows (score_metric = "accuracy" or "f1_macro").
- Skip a (task, target, variant) silently if fewer than 50 finite labels.
- Set determinism: torch.manual_seed(args.seed), pl.seed_everything(args.seed).
"""

import argparse
from pathlib import Path

# ROCm workaround
try:
    import torch  # noqa: F401
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "preferred_blas_library"):
        torch.backends.cuda.preferred_blas_library("hipblas")
except Exception:
    pass


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--embeddings", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-epochs", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    raise NotImplementedError(
        "TODO(engaging): implement MLP-probe sweep. See module docstring for the contract.\n"
        f"Args: {vars(args)}"
    )


if __name__ == "__main__":
    main()
