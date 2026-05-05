"""
Extended AION encoder for HSC and Legacy galleries.

Supports custom source_type filtering via --source-types. Defaults:
  hsc    → source_type∈{0,1}  (~366k rows)
  legacy → source_type∈{0,2}  (~205k rows)

Use --source-types 0 for overlap-only (source_type=0, ~103k rows per survey).
Use --suffix to name the output file: anomaly_latents_aion_{suffix}.h5

Run from galaxy_model/:
  # HSC-only experiment gallery (all HSC-valid):
  python anomaly_detection/encode_latents_aion_extended.py --survey hsc --suffix hsc_extended
  # Combined experiment galleries (overlap only):
  python anomaly_detection/encode_latents_aion_extended.py --survey hsc    --source-types 0 --suffix overlap_hsc
  python anomaly_detection/encode_latents_aion_extended.py --survey legacy --source-types 0 --suffix overlap_legacy
"""
import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path

_here = Path(__file__).resolve().parent
_src = _here.parent
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

import h5py
import numpy as np
import torch
torch.backends.cuda.preferred_blas_library("hipblas")
from tqdm import tqdm

from aion import AION
from aion.codecs import CodecManager
from aion.modalities import HSCImage, LegacySurveyImage

NEIGHBORS_HDF5 = "/work1/jeroenaudenaert/pablomer/data/neighbours_v2.h5"
OUTPUT_DIR = _here / "outputs"

_SURVEY_CFG = {
    "hsc": {
        "source_types": [0, 1],
        "img_key": "images_hsc",
        "modality_cls": HSCImage,
        "bands": ["HSC-G", "HSC-R", "HSC-I", "HSC-Z", "HSC-Y"],
        "emb_key": "embeddings_mean_hsc",
    },
    "legacy": {
        "source_types": [0, 2],
        "img_key": "images_legacy",
        "modality_cls": LegacySurveyImage,
        "bands": ["DES-G", "DES-R", "DES-I", "DES-Z"],
        "emb_key": "embeddings_mean_legacy",
    },
}


def encode(survey: str, source_types: list, suffix: str, batch_size: int, n_max, device_str: str):
    cfg = dict(_SURVEY_CFG[survey])  # copy so we can override source_types
    cfg["source_types"] = source_types
    device = torch.device(device_str)

    print(f"Loading AION model (polymathic-ai/aion-base) for survey={survey} ...")
    model = AION.from_pretrained("polymathic-ai/aion-base").to(device)
    codec_manager = CodecManager(device=device_str)
    model.eval()

    with h5py.File(NEIGHBORS_HDF5, "r") as f:
        indexes = np.where(np.isin(f["source_type"][:], source_types))[0]

    n_total = len(indexes)
    n_use = min(n_max, n_total) if n_max is not None else n_total
    indexes = indexes[:n_use]
    print(f"Encoding {n_use} examples (total valid: {n_total}) — survey: {survey.upper()}, "
          f"source_types={source_types}")

    all_embeddings, all_raw_index = [], []

    with h5py.File(NEIGHBORS_HDF5, "r") as f:
        for start in tqdm(range(0, n_use, batch_size), desc=f"Encoding AION ({survey})"):
            end = min(start + batch_size, n_use)
            indices = indexes[start:end]
            img_tensor = torch.from_numpy(f[cfg["img_key"]][indices]).to(device)
            image = cfg["modality_cls"](flux=img_tensor, bands=cfg["bands"])
            tokens = codec_manager.encode(image)
            with torch.no_grad():
                emb = model.encode(tokens)
            all_embeddings.append(emb.mean(dim=1).cpu().numpy().astype(np.float32))
            all_raw_index.append(indices.astype(np.int64))

    embeddings = np.concatenate(all_embeddings, axis=0)
    raw_index = np.concatenate(all_raw_index, axis=0)
    print(f"  {cfg['emb_key']}: {embeddings.shape}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"anomaly_latents_aion_{suffix}.h5"
    tmp_fd, tmp_path = tempfile.mkstemp(
        suffix=".h5", prefix=f"aion_{suffix}_", dir=tempfile.gettempdir()
    )
    os.close(tmp_fd)
    try:
        with h5py.File(tmp_path, "w") as f:
            f.create_dataset("raw_index", data=raw_index, compression="gzip", compression_opts=4)
            f.create_dataset(cfg["emb_key"], data=embeddings, compression="gzip", compression_opts=4)
            f.attrs["n_use"] = n_use
            f.attrs["survey"] = survey
            f.attrs["source_types"] = source_types
            f.attrs["embedding_dim"] = embeddings.shape[1]
        shutil.move(tmp_path, out_path)
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise

    print(f"Saved: {out_path}")
    return out_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--survey", choices=["hsc", "legacy"], required=True)
    p.add_argument("--source-types", nargs="+", type=int, default=None,
                   help="source_type values to include (default: [0,1] for hsc, [0,2] for legacy)")
    p.add_argument("--suffix", default=None,
                   help="Output file suffix: anomaly_latents_aion_{suffix}.h5 "
                        "(default: {survey}_extended)")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--n-max", type=int, default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    default_source_types = _SURVEY_CFG[args.survey]["source_types"]
    source_types = args.source_types if args.source_types is not None else default_source_types
    suffix = args.suffix if args.suffix is not None else f"{args.survey}_extended"
    encode(args.survey, source_types, suffix, args.batch_size, args.n_max, args.device)


if __name__ == "__main__":
    main()
