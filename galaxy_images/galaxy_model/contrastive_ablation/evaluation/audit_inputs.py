#!/usr/bin/env python3
"""Audit all inputs required by the resolution-collapse evaluation.

The command is intentionally read-only unless ``--out`` is supplied. It checks
paths, fixed overlap identifiers, embedding keys/dimensions, and checkpoint
availability without loading image binaries into memory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / "evaluation_config.json"
REQUIRED_EMBEDDING_KEYS = (
    "overlap/hsc_e1",
    "overlap/legacy_e1",
    "overlap/combined_e1",
)


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def load_config(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        return json.load(handle)


def jsonable(value: Any) -> Any:
    """Convert common HDF5/numpy attribute values without assuming scalars."""
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list):
        return [jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def inspect_parquet(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {"path": str(path), "exists": path.is_file()}
    if not path.is_file():
        return result
    try:
        import pyarrow.parquet as pq

        parquet = pq.ParquetFile(path)
        result.update(
            rows=parquet.metadata.num_rows,
            row_groups=parquet.metadata.num_row_groups,
            columns=parquet.schema_arrow.names,
            sha256=file_sha256(path),
        )
    except Exception as exc:  # dependency or corrupt input; report, do not hide
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def inspect_h5(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {"path": str(path), "exists": path.is_file()}
    if not path.is_file():
        return result
    try:
        import h5py

        with h5py.File(path, "r") as handle:
            result["attrs"] = {
                str(key): jsonable(value)
                for key, value in handle.attrs.items()
            }
            datasets: dict[str, Any] = {}
            for key in REQUIRED_EMBEDDING_KEYS:
                datasets[key] = (
                    {"shape": list(handle[key].shape), "dtype": str(handle[key].dtype)}
                    if key in handle
                    else None
                )
            result["datasets"] = datasets
            shapes = [datasets[key]["shape"] for key in REQUIRED_EMBEDDING_KEYS if datasets[key]]
            if len(shapes) == len(REQUIRED_EMBEDDING_KEYS):
                n_hsc, n_legacy, n_combined = (shape[0] for shape in shapes)
                hsc_dim, legacy_dim, combined_dim = (shape[1] for shape in shapes)
                result["valid_alignment"] = n_hsc == n_legacy == n_combined
                result["valid_combined_dim"] = combined_dim == hsc_dim + legacy_dim
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def checkpoint_candidates(root: Path, variant: str) -> list[str]:
    names = [variant]
    if variant.startswith("contrastive-"):
        names.append(variant.removeprefix("contrastive-"))
    else:
        names.append(f"contrastive-{variant}")
    matches: set[Path] = set()
    for name in names:
        directory = root / name
        if directory.is_dir():
            matches.update(directory.glob("best-*.ckpt"))
            matches.update(directory.glob("*.ckpt"))
    return [str(path) for path in sorted(matches)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    paths = {name: Path(value) for name, value in config["paths"].items()}
    variants = {
        config["primary_pair"]["ours"],
        config["primary_pair"]["contrastive"],
        config["secondary_pair"]["ours"],
        config["secondary_pair"]["contrastive"],
        *config.get("contrastive_mechanism_arms", []),
    }

    report: dict[str, Any] = {
        "config": str(args.config),
        "metric_version": config["metric_version"],
        "overlap_index": inspect_parquet(paths["index"]),
        "catalogs": {
            "hsc": inspect_parquet(paths["hsc_dir"] / "catalog.parquet"),
            "legacy": inspect_parquet(paths["legacy_dir"] / "catalog.parquet"),
        },
        "image_binaries": {
            "hsc": {
                "path": str(paths["hsc_dir"] / "hsc_flux.bin"),
                "exists": (paths["hsc_dir"] / "hsc_flux.bin").is_file(),
            },
            "legacy": {
                "path": str(paths["legacy_dir"] / "legacy_flux.bin"),
                "exists": (paths["legacy_dir"] / "legacy_flux.bin").is_file(),
            },
        },
        "variants": {},
    }

    for variant in sorted(variants):
        embedding = paths["embeddings_dir"] / f"{variant}.h5"
        report["variants"][variant] = {
            "embedding": inspect_h5(embedding),
            "incoming_checkpoints": checkpoint_candidates(
                paths["incoming_checkpoints_dir"], variant
            ),
            "canonical_checkpoints": checkpoint_candidates(
                paths["canonical_checkpoints_dir"], variant
            ),
        }

    index_rows = report["overlap_index"].get("rows")
    for variant_data in report["variants"].values():
        datasets = variant_data["embedding"].get("datasets", {})
        hsc = datasets.get("overlap/hsc_e1")
        if index_rows is not None and hsc is not None:
            variant_data["embedding"]["matches_index_rows"] = hsc["shape"][0] == index_rows

    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
