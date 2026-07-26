#!/usr/bin/env python3
"""Audit catalog schemas and build a paired, model-independent quality table."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from common import (
    BANDS,
    METRIC_VERSION,
    band_column,
    infer_column,
    normalize_name,
    normalized_id,
    require_file,
    robust_rank,
    scalar_or_vector,
    validate_unique,
    write_json,
)

FWHM_PER_SIGMA = 2.0 * math.sqrt(2.0 * math.log(2.0))

# HSC releases use several spellings for the same HSM/adaptive-moment fields.
MOMENT_ALIASES = {
    "obj_xx": (
        "sdssshape_shape11",
        "shape_hsm_moments_11",
        "hsm_moments_11",
        "adaptive_moments_xx",
        "moments_xx",
    ),
    "obj_yy": (
        "sdssshape_shape22",
        "shape_hsm_moments_22",
        "hsm_moments_22",
        "adaptive_moments_yy",
        "moments_yy",
    ),
    "obj_xy": (
        "sdssshape_shape12",
        "shape_hsm_moments_12",
        "hsm_moments_12",
        "adaptive_moments_xy",
        "moments_xy",
    ),
    "psf_xx": (
        "sdssshape_psf_shape11",
        "shape_hsm_psfmoments_11",
        "shape_hsm_psf_moments_11",
        "psf_hsm_moments_11",
        "psf_adaptive_moments_xx",
        "psf_moments_xx",
    ),
    "psf_yy": (
        "sdssshape_psf_shape22",
        "shape_hsm_psfmoments_22",
        "shape_hsm_psf_moments_22",
        "psf_hsm_moments_22",
        "psf_adaptive_moments_yy",
        "psf_moments_yy",
    ),
    "psf_xy": (
        "sdssshape_psf_shape12",
        "shape_hsm_psfmoments_12",
        "shape_hsm_psf_moments_12",
        "psf_hsm_moments_12",
        "psf_adaptive_moments_xy",
        "psf_moments_xy",
    ),
}
HSC_FLUX = ("cmodel_flux", "forced_cmodel_flux", "flux", "model_flux")
HSC_FLUX_ERR = (
    "cmodel_fluxerr",
    "cmodel_flux_err",
    "cmodel_flux_error",
    "cmodel_flux_sigma",
    "forced_cmodel_flux_err",
    "flux_err",
    "flux_error",
    "flux_sigma",
)
HSC_PSF_FWHM = ("image_psf_fwhm", "psf_fwhm", "seeing_fwhm")
HSC_IMAGE_SCALE = ("image_scale", "pixel_scale", "pixel_scale_arcsec")


def _read(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    require_file(path, "parquet")
    return pd.read_parquet(path, columns=columns)


def _schema(path: Path) -> dict:
    require_file(path, "parquet")
    schema = pq.read_schema(path)
    return {field.name: str(field.type) for field in schema}


def _moment_series(df: pd.DataFrame, kind: str, band: str, bi: int):
    series, source = scalar_or_vector(df, MOMENT_ALIASES[kind], band, bi)
    if series is not None:
        return series, source

    # Last-resort conservative token discovery. Never silently choose ambiguity.
    is_psf = kind.startswith("psf")
    axis = kind.split("_")[1]
    axis_aliases = {"xx": ("xx", "11"), "yy": ("yy", "22"), "xy": ("xy", "12")}[axis]
    hits = []
    for col in df.columns:
        tokens = normalize_name(col).split("_")
        if band not in tokens:
            continue
        if not any(a in tokens for a in axis_aliases):
            continue
        if not (
            "moment" in tokens
            or "moments" in tokens
            or "hsm" in tokens
            or "sdssshape" in tokens
        ):
            continue
        has_psf = "psf" in tokens or any(t.startswith("psf") for t in tokens)
        if has_psf == is_psf:
            hits.append(col)
    if len(hits) == 1:
        return pd.to_numeric(df[hits[0]], errors="coerce"), hits[0]
    return None, None


def discover(hsc: pd.DataFrame, legacy: pd.DataFrame) -> dict:
    found: dict[str, dict] = {"hsc": {}, "legacy": {}}
    for bi, band in enumerate(BANDS):
        b: dict[str, str | None] = {}
        for kind in MOMENT_ALIASES:
            _, b[kind] = _moment_series(hsc, kind, band, bi)
        _, b["flux"] = scalar_or_vector(hsc, HSC_FLUX, band, bi)
        _, b["flux_err"] = scalar_or_vector(hsc, HSC_FLUX_ERR, band, bi)
        _, b["psf_fwhm"] = scalar_or_vector(hsc, HSC_PSF_FWHM, band, bi)
        _, b["image_scale"] = scalar_or_vector(hsc, HSC_IMAGE_SCALE, band, bi)
        found["hsc"][band] = b

        found["legacy"][band] = {
            "psf_fwhm": band_column(legacy.columns, "PSFSIZE", band),
            "flux": band_column(legacy.columns, "FLUX", band),
            "flux_ivar": band_column(legacy.columns, "FLUX_IVAR", band),
        }
    return found


def audit_report(overlap_path: Path, hsc_path: Path, legacy_path: Path) -> dict:
    schemas = {
        "overlap": _schema(overlap_path),
        "hsc": _schema(hsc_path),
        "legacy": _schema(legacy_path),
    }
    hsc = pd.DataFrame(columns=list(schemas["hsc"]))
    legacy = pd.DataFrame(columns=list(schemas["legacy"]))
    discovered = discover(hsc, legacy)
    required_overlap = {
        "hsc_image_idx",
        "legacy_image_idx",
        "hsc_object_id",
        "legacy_object_id",
    }
    missing_overlap = sorted(required_overlap - set(schemas["overlap"]))
    missing_metrics = []
    for survey, bands in discovered.items():
        for band, fields in bands.items():
            for field, source in fields.items():
                if source is None and field != "image_scale":
                    missing_metrics.append(f"{survey}.{band}.{field}")
    return {
        "paths": {
            "overlap": str(overlap_path),
            "hsc": str(hsc_path),
            "legacy": str(legacy_path),
        },
        "row_counts": {
            "overlap": pq.ParquetFile(overlap_path).metadata.num_rows,
            "hsc": pq.ParquetFile(hsc_path).metadata.num_rows,
            "legacy": pq.ParquetFile(legacy_path).metadata.num_rows,
        },
        "schemas": schemas,
        "discovered": discovered,
        "missing_overlap_columns": missing_overlap,
        "missing_metric_fields": missing_metrics,
        "ok": not missing_overlap and not missing_metrics,
    }


def _join_catalog(overlap: pd.DataFrame, catalog: pd.DataFrame, survey: str) -> pd.DataFrame:
    overlap_key = f"{survey}_image_idx"
    if "image_idx" not in catalog:
        raise ValueError(f"{survey} catalog has no image_idx column")
    validate_unique(catalog, "image_idx", f"{survey} catalog")
    renamed = catalog.rename(columns={c: f"{survey}__{c}" for c in catalog if c != "image_idx"})
    out = overlap.merge(
        renamed,
        left_on=overlap_key,
        right_on="image_idx",
        how="left",
        validate="many_to_one",
        indicator=f"_{survey}_join",
    ).drop(columns=["image_idx"])
    if (out[f"_{survey}_join"] != "both").any():
        n = int((out[f"_{survey}_join"] != "both").sum())
        raise ValueError(f"{n} overlap rows did not join to the {survey} catalog")
    return out.drop(columns=[f"_{survey}_join"])


def _prefixed_frame(joined: pd.DataFrame, survey: str) -> pd.DataFrame:
    prefix = f"{survey}__"
    return joined[[c for c in joined if c.startswith(prefix)]].rename(
        columns=lambda c: c[len(prefix) :]
    )


def _positive_definite(xx, yy, xy):
    return (xx > 0) & (yy > 0) & ((xx * yy - xy * xy) > 0)


def _band_metrics(
    hsc: pd.DataFrame,
    legacy: pd.DataFrame,
    band: str,
    bi: int,
    pixel_scale: float,
    moments_unit: str,
) -> tuple[dict[str, pd.Series], dict]:
    src = {}
    vals = {}
    for kind in MOMENT_ALIASES:
        vals[kind], src[kind] = _moment_series(hsc, kind, band, bi)
    hflux, src["hsc_flux"] = scalar_or_vector(hsc, HSC_FLUX, band, bi)
    herr, src["hsc_flux_err"] = scalar_or_vector(hsc, HSC_FLUX_ERR, band, bi)
    hpsf, src["hsc_psf_fwhm"] = scalar_or_vector(hsc, HSC_PSF_FWHM, band, bi)
    lpsf_col = band_column(legacy.columns, "PSFSIZE", band)
    lflux_col = band_column(legacy.columns, "FLUX", band)
    livar_col = band_column(legacy.columns, "FLUX_IVAR", band)
    src.update(
        legacy_psf_fwhm=lpsf_col,
        legacy_flux=lflux_col,
        legacy_flux_ivar=livar_col,
    )
    if any(v is None for v in src.values()):
        missing = [k for k, v in src.items() if v is None]
        raise ValueError(f"band {band}: unresolved required fields: {missing}")

    image_scale, src["hsc_image_scale"] = scalar_or_vector(
        hsc, HSC_IMAGE_SCALE, band, bi
    )
    if moments_unit == "pixel2":
        if image_scale is None:
            scale = pd.Series(pixel_scale, index=hsc.index, dtype=float)
            src["hsc_image_scale"] = f"CLI fallback ({pixel_scale} arcsec/pixel)"
        else:
            scale = pd.to_numeric(image_scale, errors="coerce").where(
                pd.to_numeric(image_scale, errors="coerce") > 0, pixel_scale
            )
    else:
        scale = pd.Series(1.0, index=hsc.index)
    oxx, oyy, oxy = vals["obj_xx"], vals["obj_yy"], vals["obj_xy"]
    pxx, pyy, pxy = vals["psf_xx"], vals["psf_yy"], vals["psf_xy"]
    obs_valid = _positive_definite(oxx, oyy, oxy)
    psf_valid = _positive_definite(pxx, pyy, pxy)
    cxx, cyy, cxy = oxx - pxx, oyy - pyy, oxy - pxy
    corr_valid = obs_valid & psf_valid & _positive_definite(cxx, cyy, cxy)
    sigma_corr = pd.Series(np.where(corr_valid, np.power(cxx * cyy - cxy * cxy, 0.25), np.nan))
    sigma_psf_mom = pd.Series(np.where(psf_valid, np.power(pxx * pyy - pxy * pxy, 0.25), np.nan))
    size_fwhm = FWHM_PER_SIGMA * sigma_corr * scale
    size_proxy = sigma_corr / sigma_psf_mom

    hpsf = pd.to_numeric(hpsf, errors="coerce")
    lpsf = pd.to_numeric(legacy[lpsf_col], errors="coerce")
    herr_num = pd.to_numeric(herr, errors="coerce")
    hsnr = pd.to_numeric(hflux, errors="coerce") / herr_num
    livar = pd.to_numeric(legacy[livar_col], errors="coerce")
    lsnr = pd.to_numeric(legacy[lflux_col], errors="coerce") * np.sqrt(livar.clip(lower=0))
    hsnr = hsnr.where((herr_num > 0) & (hsnr > 0))
    lsnr = lsnr.where((livar > 0) & (lsnr > 0))
    hres = (size_fwhm / hpsf).where((size_fwhm > 0) & (hpsf > 0))
    lres = (size_fwhm / lpsf).where((size_fwhm > 0) & (lpsf > 0))
    return {
        f"hsc_size_proxy_{band}": size_proxy.where(corr_valid),
        f"hsc_size_fwhm_{band}": size_fwhm.where(corr_valid),
        f"hsc_psf_fwhm_{band}": hpsf.where(hpsf > 0),
        f"legacy_psf_fwhm_{band}": lpsf.where(lpsf > 0),
        f"hsc_resolution_{band}": hres,
        f"legacy_resolution_{band}": lres,
        f"resolution_gap_{band}": hres - lres,
        f"resolution_log_ratio_{band}": np.log(hres) - np.log(lres),
        f"hsc_snr_{band}": hsnr,
        f"legacy_snr_{band}": lsnr,
        f"snr_gap_{band}": np.log1p(hsnr) - np.log1p(lsnr),
        f"moments_psf_corrected_valid_{band}": corr_valid.astype(bool),
    }, src


def _median(df: pd.DataFrame, prefix: str) -> pd.Series:
    return df[[f"{prefix}_{b}" for b in BANDS]].median(axis=1, skipna=True)


def build_quality_table(
    overlap_path: Path,
    hsc_path: Path,
    legacy_path: Path,
    *,
    pixel_scale: float,
    moments_unit: str,
    metric_version: str,
) -> tuple[pd.DataFrame, dict]:
    overlap = _read(overlap_path)
    required = {"hsc_image_idx", "legacy_image_idx", "hsc_object_id", "legacy_object_id"}
    missing = required - set(overlap)
    if missing:
        raise ValueError(f"overlap is missing required columns: {sorted(missing)}")
    if overlap.empty:
        raise ValueError("overlap.parquet is empty")

    # Discover from cheap schemas, then read only selected columns from the large catalogs.
    hsc_columns = list(_schema(hsc_path))
    legacy_columns = list(_schema(legacy_path))
    discovered = discover(
        pd.DataFrame(columns=hsc_columns), pd.DataFrame(columns=legacy_columns)
    )
    missing_metrics = [
        f"{survey}.{band}.{field}"
        for survey, bands in discovered.items()
        for band, fields in bands.items()
        for field, source in fields.items()
        if source is None and field != "image_scale"
    ]
    if missing_metrics:
        raise ValueError(
            "required catalog fields could not be discovered: "
            + ", ".join(missing_metrics)
            + "; run the audit subcommand for the full schema report"
        )

    def selected(survey: str) -> list[str]:
        columns = {"image_idx"}
        for fields in discovered[survey].values():
            for source in fields.values():
                if source:
                    columns.add(source.split("[", 1)[0])
        return sorted(columns)

    hsc_cat = _read(hsc_path, selected("hsc"))
    legacy_cat = _read(legacy_path, selected("legacy"))
    joined = _join_catalog(overlap, hsc_cat, "hsc")
    joined = _join_catalog(joined, legacy_cat, "legacy")
    hsc, legacy = _prefixed_frame(joined, "hsc"), _prefixed_frame(joined, "legacy")

    out = overlap.copy()
    out["hsc_object_id"] = normalized_id(out["hsc_object_id"]).to_numpy()
    out["legacy_object_id"] = normalized_id(out["legacy_object_id"]).to_numpy()
    sources = {}
    for bi, band in enumerate(BANDS):
        metrics, sources[band] = _band_metrics(
            hsc, legacy, band, bi, pixel_scale, moments_unit
        )
        for name, values in metrics.items():
            out[name] = np.asarray(values)

    for prefix in (
        "hsc_size_proxy",
        "hsc_size_fwhm",
        "hsc_psf_fwhm",
        "legacy_psf_fwhm",
        "hsc_resolution",
        "legacy_resolution",
        "resolution_gap",
        "resolution_log_ratio",
        "hsc_snr",
        "legacy_snr",
        "snr_gap",
    ):
        out[prefix] = _median(out, prefix)

    out["n_resolution_bands"] = out[[f"resolution_gap_{b}" for b in BANDS]].notna().sum(axis=1)
    out["n_snr_bands"] = out[[f"snr_gap_{b}" for b in BANDS]].notna().sum(axis=1)
    out["coverage_resolution"] = out["n_resolution_bands"] >= 2
    out["coverage_snr"] = out["n_snr_bands"] >= 2
    out["coverage_joint"] = out["coverage_resolution"] & out["coverage_snr"]
    # Two-sided ranks prevent a large ratio from scoring highly when both
    # surveys are poor: HSC must be good while Legacy is poor.
    out["hsc_resolution_rank"] = robust_rank(out["hsc_resolution"])
    out["legacy_unresolved_rank"] = robust_rank(-out["legacy_resolution"])
    out["resolution_contrast_rank"] = robust_rank(out["resolution_log_ratio"])
    out["resolution_rank"] = out[
        [
            "hsc_resolution_rank",
            "legacy_unresolved_rank",
            "resolution_contrast_rank",
        ]
    ].mean(axis=1).where(out["coverage_resolution"])
    out["hsc_snr_rank"] = robust_rank(np.log1p(out["hsc_snr"]))
    out["legacy_low_snr_rank"] = robust_rank(-np.log1p(out["legacy_snr"]))
    out["snr_contrast_rank"] = robust_rank(out["snr_gap"])
    out["snr_rank"] = out[
        ["hsc_snr_rank", "legacy_low_snr_rank", "snr_contrast_rank"]
    ].mean(axis=1).where(out["coverage_snr"])
    out["joint_score"] = ((out["resolution_rank"] + out["snr_rank"]) / 2).where(
        out["coverage_joint"]
    )
    out["metric_version"] = metric_version
    out["quality_rank"] = out["joint_score"].rank(ascending=False, method="min")
    out = out.sort_values(
        ["joint_score", "resolution_gap", "snr_gap"],
        ascending=False,
        na_position="last",
    ).reset_index(drop=True)

    metadata = {
        "metric_version": metric_version,
        "definition": {
            "intrinsic_size_fwhm": "2.35482 * det(M_object-M_psf)^(1/4)",
            "hsc_size_proxy": "det(M_object-M_psf)^(1/4) / det(M_psf)^(1/4)",
            "resolution": "intrinsic_size_fwhm / catalog_psf_fwhm",
            "resolution_gap": "median_griz(HSC_resolution-Legacy_resolution)",
            "resolution_log_ratio": "median_griz(log(HSC_resolution/Legacy_resolution))",
            "snr_gap": "median_griz(log1p(HSC_flux/error)-log1p(Legacy_flux*sqrt(ivar)))",
            "resolution_rank": (
                "mean(rank(high HSC resolution), rank(low Legacy resolution), "
                "rank(high HSC/Legacy resolution ratio))"
            ),
            "snr_rank": (
                "mean(rank(high HSC SNR), rank(low Legacy SNR), "
                "rank(high HSC-Legacy log-SNR gap))"
            ),
            "joint_score": "mean(resolution_rank, snr_rank)",
        },
        "moments_unit": moments_unit,
        "hsc_pixel_scale_arcsec": pixel_scale,
        "sources": sources,
        "coverage": {
            key: int(out[key].sum())
            for key in ("coverage_resolution", "coverage_snr", "coverage_joint")
        },
        "rows": len(out),
    }
    return out, metadata


def _paths(args):
    return (
        require_file(args.overlap, "fixed overlap.parquet"),
        require_file(args.hsc_catalog, "HSC catalog"),
        require_file(args.legacy_catalog, "Legacy catalog"),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    for name in ("audit", "build"):
        p = sub.add_parser(name)
        p.add_argument("--overlap", required=True, type=Path)
        p.add_argument("--hsc-catalog", required=True, type=Path)
        p.add_argument("--legacy-catalog", required=True, type=Path)
        p.add_argument("--report", type=Path, help="write JSON schema/provenance report")
        if name == "build":
            p.add_argument("--out", required=True, type=Path)
            p.add_argument("--candidates", type=Path)
            p.add_argument("--top-n", type=int, default=200)
            p.add_argument("--hsc-pixel-scale", type=float, default=0.168)
            p.add_argument("--moments-unit", choices=("pixel2", "arcsec2"), default="pixel2")
            p.add_argument("--metric-version", default=METRIC_VERSION)
    args = ap.parse_args()
    overlap, hsc, legacy = _paths(args)

    if args.command == "audit":
        report = audit_report(overlap, hsc, legacy)
        print(json.dumps(report, indent=2))
        if args.report:
            write_json(args.report, report)
        if not report["ok"]:
            raise SystemExit(
                "schema audit failed; inspect missing_metric_fields. "
                "No heuristic was allowed to choose an ambiguous column."
            )
        return

    table, metadata = build_quality_table(
        overlap,
        hsc,
        legacy,
        pixel_scale=args.hsc_pixel_scale,
        moments_unit=args.moments_unit,
        metric_version=args.metric_version,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(args.out, index=False)
    candidates = args.candidates or args.out.with_name("quality_candidates_top200.csv")
    keep = [
        "quality_rank",
        "hsc_object_id",
        "legacy_object_id",
        "hsc_image_idx",
        "legacy_image_idx",
        "ra",
        "dec",
        "resolution_gap",
        "snr_gap",
        "joint_score",
        "n_resolution_bands",
        "n_snr_bands",
        "coverage_joint",
        "metric_version",
    ]
    keep = [c for c in keep if c in table]
    table.loc[table["coverage_joint"], keep].head(args.top_n).to_csv(candidates, index=False)
    metadata["outputs"] = {"quality_table": str(args.out), "candidates": str(candidates)}
    report_path = args.report or args.out.with_suffix(".metadata.json")
    write_json(report_path, metadata)
    print(f"wrote {args.out} ({len(table)} rows)")
    print(f"wrote {candidates} ({min(args.top_n, int(table.coverage_joint.sum()))} rows)")
    print(f"wrote {report_path}")


if __name__ == "__main__":
    main()
