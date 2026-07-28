"""Shared helpers for the quality-gallery and adversarial evaluation CLIs."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

BANDS = ("g", "r", "i", "z")
METRIC_VERSION = "hsc_detmom_psfcorr__griz_twosided_resolution_snr__rankmean_v2_candidate"


def require_file(path: Path, label: str) -> Path:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def normalize_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_")


def finite(values) -> np.ndarray:
    return np.isfinite(np.asarray(values, dtype=float))


def robust_rank(values: pd.Series) -> pd.Series:
    """Percentile rank in [0, 1], preserving missing values."""
    x = pd.to_numeric(values, errors="coerce")
    return x.rank(method="average", pct=True)


def normalized_id(values: Iterable) -> pd.Series:
    """Normalize byte/string/integer object IDs without changing their identity."""
    def one(x):
        if isinstance(x, bytes):
            x = x.decode("utf-8", errors="replace")
        s = str(x).strip()
        if (s.startswith("b'") and s.endswith("'")) or (
            s.startswith('b"') and s.endswith('"')
        ):
            s = s[2:-1]
        return s

    return pd.Series([one(x) for x in values], dtype="string")


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def dataframe_sha256(df: pd.DataFrame, columns: list[str]) -> str:
    # Cast BEFORE filling: on nullable dtypes (Int64/boolean) `.fillna("<NA>")`
    # raises TypeError because the sentinel is not a valid value for the dtype.
    # `.astype("string")` maps NaN and pd.NA alike to pd.NA, so one fillna covers both.
    payload = df[columns].astype("string").fillna("<NA>").to_csv(index=False).encode()
    return hashlib.sha256(payload).hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def infer_column(
    columns: Iterable[str],
    *,
    exact: Iterable[str] = (),
    required_tokens: Iterable[str] = (),
    forbidden_tokens: Iterable[str] = (),
) -> str | None:
    """Conservative schema discovery: exact aliases first, then unique token match."""
    names = list(columns)
    norm = {name: normalize_name(name) for name in names}
    exact_norm = {normalize_name(x) for x in exact}
    hits = [name for name in names if norm[name] in exact_norm]
    if len(hits) == 1:
        return hits[0]
    req = tuple(normalize_name(x) for x in required_tokens)
    bad = tuple(normalize_name(x) for x in forbidden_tokens)
    hits = [
        name
        for name in names
        if all(token in norm[name].split("_") for token in req)
        and not any(token in norm[name].split("_") for token in bad)
    ]
    return hits[0] if len(hits) == 1 else None


def band_column(columns: Iterable[str], stem: str, band: str) -> str | None:
    """Find scalar columns such as FLUX_G, g_cmodel_flux, or flux_g."""
    names = list(columns)
    stem_tokens = normalize_name(stem).split("_")
    exact_forms = {
        "_".join(stem_tokens + [band]),
        "_".join([band] + stem_tokens),
    }
    exact = [name for name in names if normalize_name(name) in exact_forms]
    if len(exact) == 1:
        return exact[0]
    candidates = []
    for name in names:
        tokens = normalize_name(name).split("_")
        # Do not let FLUX match FLUX_IVAR or FLUX_ERR.
        extras = set(tokens) - set(stem_tokens) - {band}
        if band in tokens and all(t in tokens for t in stem_tokens) and not extras:
            candidates.append(name)
    return candidates[0] if len(candidates) == 1 else None


def scalar_or_vector(
    df: pd.DataFrame,
    stem_aliases: Iterable[str],
    band: str,
    band_index: int,
) -> tuple[pd.Series | None, str | None]:
    """Resolve a per-band scalar or a griz(/y) list-valued catalog column."""
    for stem in stem_aliases:
        col = band_column(df.columns, stem, band)
        if col:
            return pd.to_numeric(df[col], errors="coerce"), col
    aliases = {normalize_name(x) for x in stem_aliases}
    vector_hits = [
        c
        for c in df.columns
        if normalize_name(c) in aliases
        or any(normalize_name(c).endswith("_" + alias) for alias in aliases)
    ]
    if len(vector_hits) != 1:
        return None, None
    col = vector_hits[0]

    def get(v):
        try:
            return float(v[band_index])
        except (IndexError, TypeError, ValueError):
            return np.nan

    return df[col].map(get).astype(float), f"{col}[{band_index}]"


def validate_unique(df: pd.DataFrame, column: str, label: str) -> None:
    if column not in df:
        raise ValueError(f"{label} is missing required key column {column!r}")
    dup = int(df[column].duplicated().sum())
    if dup:
        raise ValueError(f"{label}.{column} has {dup} duplicate keys; join is ambiguous")
