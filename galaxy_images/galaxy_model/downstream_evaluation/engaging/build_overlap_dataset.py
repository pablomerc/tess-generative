"""
Build the index parquets for the engaging unified downstream eval.

Produces three deterministic, capped subsets of galaxy indices:

  - overlap.parquet      HSC ∩ Legacy galaxies (matched within --match-radius-arcsec)
  - hsc_only.parquet     HSC galaxies NOT in the overlap
  - legacy_only.parquet  Legacy galaxies NOT in the overlap

Each subset is randomly sampled (seeded) and capped at --max-per-subset rows.
A full-overlap parquet (overlap_full.parquet, before sampling) is also written
for diagnostics.

Crossmatch is sky-coordinate based (astropy SkyCoord.match_to_catalog_sky),
since HSC object_id is int64 and Legacy object_id is string with no shared key.

Open caveat about the HSC binary
--------------------------------
hsc_flux.bin only contains a subset of the HSC catalog rows (file size implies
~57.5k of the 474k catalog rows). This script derives N_actual from the file
size and clips HSC indices to image_idx < N_actual before sampling, so the
downstream embedding stage never tries to read past the binary.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from astropy.coordinates import SkyCoord
from astropy import units as u

# Per-survey image-binary geometry. Used to detect the row-count actually
# present in each .bin (HSC ships partial; Legacy ships full).
HSC_BANDS, HSC_H, HSC_W = 5, 160, 160
HSC_BYTES_PER_GALAXY = HSC_BANDS * HSC_H * HSC_W * 2  # float16

LEGACY_BANDS, LEGACY_H, LEGACY_W = 4, 160, 160
LEGACY_BYTES_PER_GALAXY = LEGACY_BANDS * LEGACY_H * LEGACY_W * 2  # float16


def detect_n_in_bin(bin_path: Path, bytes_per_galaxy: int) -> int:
    sz = os.path.getsize(bin_path)
    n = sz // bytes_per_galaxy
    if n * bytes_per_galaxy != sz:
        print(
            f"  [warn] {bin_path.name} size {sz} is not a multiple of "
            f"{bytes_per_galaxy} bytes/galaxy — last {sz - n*bytes_per_galaxy} "
            f"bytes will be ignored (truncated record).",
            file=sys.stderr,
        )
    return int(n)


def load_radec(catalog_path: Path) -> pd.DataFrame:
    """Read just (image_idx, object_id, ra, dec) from a catalog parquet."""
    cols = ["image_idx", "object_id", "ra", "dec"]
    df = pq.read_table(catalog_path, columns=cols).to_pandas()
    return df


def crossmatch(
    hsc: pd.DataFrame, legacy: pd.DataFrame, radius_arcsec: float
) -> pd.DataFrame:
    """
    Match each HSC source to its nearest Legacy source on the sky. Keep
    matches within `radius_arcsec`. Returns a dataframe with columns:
        hsc_image_idx, hsc_object_id, legacy_image_idx, legacy_object_id,
        ra, dec, sep_arcsec
    Rows are unique by hsc_image_idx (each HSC row matches at most one Legacy
    row); a Legacy row may be matched by multiple HSC rows in principle, but
    with sub-arcsec radii this is essentially never the case.
    """
    hsc_sc = SkyCoord(ra=hsc["ra"].values * u.deg, dec=hsc["dec"].values * u.deg)
    leg_sc = SkyCoord(ra=legacy["ra"].values * u.deg, dec=legacy["dec"].values * u.deg)

    # For each HSC source, nearest Legacy source.
    idx_leg, sep, _ = hsc_sc.match_to_catalog_sky(leg_sc)
    sep_arcsec = sep.to(u.arcsec).value

    keep = sep_arcsec <= radius_arcsec
    hsc_kept = hsc.iloc[np.where(keep)[0]].reset_index(drop=True)
    leg_kept = legacy.iloc[idx_leg[keep]].reset_index(drop=True)

    out = pd.DataFrame(
        {
            "hsc_image_idx": hsc_kept["image_idx"].astype(np.int64).values,
            "hsc_object_id": hsc_kept["object_id"].values,
            "legacy_image_idx": leg_kept["image_idx"].astype(np.int64).values,
            "legacy_object_id": leg_kept["object_id"].astype(str).values,
            "ra": hsc_kept["ra"].values,
            "dec": hsc_kept["dec"].values,
            "sep_arcsec": sep_arcsec[keep],
        }
    )
    return out


def sample_capped(rng: np.random.Generator, df: pd.DataFrame, cap: int) -> pd.DataFrame:
    if len(df) <= cap:
        return df.reset_index(drop=True)
    sel = rng.choice(len(df), size=cap, replace=False)
    sel.sort()  # keep row order roughly along the catalog (helps locality on .bin reads)
    return df.iloc[sel].reset_index(drop=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--hsc-dir",
        type=Path,
        default=Path("/home/pablomer/orcd/scratch/hsc_downstream"),
    )
    p.add_argument(
        "--legacy-dir",
        type=Path,
        default=Path("/home/pablomer/orcd/scratch/legacy_downstream_full/full_1M"),
    )
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--max-per-subset", type=int, default=16000)
    p.add_argument("--match-radius-arcsec", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    hsc_cat = args.hsc_dir / "catalog.parquet"
    leg_cat = args.legacy_dir / "catalog.parquet"
    hsc_bin = args.hsc_dir / "hsc_flux.bin"
    leg_bin = args.legacy_dir / "legacy_flux.bin"

    print(f"Loading HSC catalog:    {hsc_cat}")
    hsc = load_radec(hsc_cat)
    print(f"Loading Legacy catalog: {leg_cat}")
    legacy = load_radec(leg_cat)
    print(f"  HSC rows:    {len(hsc):,}")
    print(f"  Legacy rows: {len(legacy):,}")

    n_hsc_bin = detect_n_in_bin(hsc_bin, HSC_BYTES_PER_GALAXY)
    n_leg_bin = detect_n_in_bin(leg_bin, LEGACY_BYTES_PER_GALAXY)
    print(f"  HSC rows in .bin:    {n_hsc_bin:,}")
    print(f"  Legacy rows in .bin: {n_leg_bin:,}")

    hsc = hsc[hsc["image_idx"] < n_hsc_bin].reset_index(drop=True)
    legacy = legacy[legacy["image_idx"] < n_leg_bin].reset_index(drop=True)
    print(f"  After clipping to .bin: HSC={len(hsc):,}  Legacy={len(legacy):,}")

    print(
        f"\nCrossmatching within {args.match_radius_arcsec}\" "
        f"(astropy SkyCoord.match_to_catalog_sky)..."
    )
    overlap_full = crossmatch(hsc, legacy, args.match_radius_arcsec)
    print(f"  matches (clipped HSC vs full Legacy): {len(overlap_full):,}")

    if len(overlap_full) == 0:
        # Common situation: hsc_flux.bin is partial (subset of HEALPix tiles).
        # Tell the user whether the *full* HSC catalog would overlap Legacy,
        # so they know whether the fix is "extend the binary" or "different sky".
        print(
            "\n  [diagnostic] zero matches inside the clipped HSC region. "
            "Re-running crossmatch against the full HSC catalog to check if "
            "any tiles outside the binary would overlap Legacy..."
        )
        full_hsc = load_radec(hsc_cat)
        unclipped = crossmatch(full_hsc, legacy, args.match_radius_arcsec)
        if len(unclipped) > 0:
            tile_counts = (
                full_hsc.merge(
                    pd.DataFrame({"image_idx": unclipped["hsc_image_idx"]}),
                    on="image_idx",
                )
                .merge(
                    pq.read_table(
                        hsc_cat, columns=["image_idx", "healpix"]
                    ).to_pandas(),
                    on="image_idx",
                )["healpix"]
                .value_counts()
                .head(10)
            )
            print(
                f"  [diagnostic] full HSC catalog has {len(unclipped):,} matches with "
                f"Legacy, but ALL of them lie outside the current hsc_flux.bin "
                f"(image_idx >= {n_hsc_bin}). Top HEALPix tiles with matches:\n"
                f"{tile_counts.to_string()}\n"
                "  Action needed: extend hsc_flux.bin to include the matching tiles "
                "(or rebuild it from full HSC PDR3) before this pipeline can produce "
                "non-empty overlap embeddings."
            )
        else:
            print(
                "  [diagnostic] full HSC catalog also has zero matches with Legacy "
                "at this radius — the two surveys cover disjoint sky in this dataset. "
                "Increase --match-radius-arcsec or use a different Legacy region."
            )
    if len(overlap_full):
        print(
            f"  separation arcsec: median={np.median(overlap_full['sep_arcsec']):.3f}  "
            f"95p={np.percentile(overlap_full['sep_arcsec'], 95):.3f}  "
            f"max={overlap_full['sep_arcsec'].max():.3f}"
        )

    overlap_full_path = args.out_dir / "overlap_full.parquet"
    overlap_full.to_parquet(overlap_full_path, index=False)
    print(f"  wrote {overlap_full_path} ({len(overlap_full):,} rows)")

    rng = np.random.default_rng(args.seed)

    overlap = sample_capped(rng, overlap_full, args.max_per_subset)
    overlap.to_parquet(args.out_dir / "overlap.parquet", index=False)
    print(f"\noverlap.parquet:      {len(overlap):,} rows (cap {args.max_per_subset})")

    in_overlap_hsc = set(overlap_full["hsc_image_idx"].tolist())
    in_overlap_leg = set(overlap_full["legacy_image_idx"].tolist())

    hsc_only_full = hsc[~hsc["image_idx"].isin(in_overlap_hsc)].reset_index(drop=True)
    leg_only_full = legacy[~legacy["image_idx"].isin(in_overlap_leg)].reset_index(
        drop=True
    )

    hsc_only = sample_capped(rng, hsc_only_full, args.max_per_subset)
    leg_only = sample_capped(rng, leg_only_full, args.max_per_subset)

    hsc_only_out = pd.DataFrame(
        {
            "hsc_image_idx": hsc_only["image_idx"].astype(np.int64).values,
            "hsc_object_id": hsc_only["object_id"].values,
            "ra": hsc_only["ra"].values,
            "dec": hsc_only["dec"].values,
        }
    )
    leg_only_out = pd.DataFrame(
        {
            "legacy_image_idx": leg_only["image_idx"].astype(np.int64).values,
            "legacy_object_id": leg_only["object_id"].astype(str).values,
            "ra": leg_only["ra"].values,
            "dec": leg_only["dec"].values,
        }
    )
    hsc_only_out.to_parquet(args.out_dir / "hsc_only.parquet", index=False)
    leg_only_out.to_parquet(args.out_dir / "legacy_only.parquet", index=False)

    print(
        f"hsc_only.parquet:     {len(hsc_only):,} rows "
        f"(cap {args.max_per_subset}, pool {len(hsc_only_full):,})"
    )
    print(
        f"legacy_only.parquet:  {len(leg_only):,} rows "
        f"(cap {args.max_per_subset}, pool {len(leg_only_full):,})"
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
