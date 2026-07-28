"""Stage 1 — encode the 103,741 anchors with a frozen contrastive dual-encoder.

Unlike the FM-decoder precomputes, this is a **self**-encoding: every latent is the
encoder's output on the anchor's *own* image. That is the semantics anomaly
detection needs (we are asking "is *this* object unusual?"), and it is what
``anomaly_detection/encode_latents_contrastive.py`` did for the AMD runs.

  physics_hsc       (N, D)  encoder_galaxy(hsc48[row])
  physics_legacy    (N, D)  encoder_galaxy(legacy48[row])
  physics_mean      (N, D)  mean of the two           <- cross-survey-averaged physics
  instrument_hsc    (N, D)  encoder_instrument(hsc48[row])
  instrument_legacy (N, D)  encoder_instrument(legacy48[row])

Both towers are read raw (``.flatten(1)``), never through the projection heads —
matching ``downstream_evaluation/engaging/prepare_combined.py`` so the anomaly
spaces are the same vectors the R^2 probes describe.

  NOTE ON THE EXISTING CF CORPUS. ``outputs/counterfactual_fm/cf_embeddings_conv1x1.h5``
  already holds conv1x1 embeddings for this corpus, but they are *counterfactual*:
  ``emb_galaxy`` is the cross-survey partner image and ``emb_instrument`` is five
  sky-neighbours, never the anchor. Reusing them would score "is this galaxy's
  neighbourhood unusual", not "is this galaxy unusual". Hence this pass.

Usage:
  python encode_latents.py --ckpt <contrastive .ckpt> --tag spatial-conv1x1
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import h5py
import numpy as np
import torch

# ROCm/AMD: prefer rocBLAS over the buggy hipBLASLt. Guarded so it is a harmless
# no-op on Engaging's NVIDIA GPUs.
try:
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "preferred_blas_library"):
        torch.backends.cuda.preferred_blas_library("hipblas")
except Exception:
    pass

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parents[3]  # .../tess-generative
for _p in (str(_PROJECT_ROOT), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from anchor_data import (  # noqa: E402
    DEFAULT_DATA_DIR,
    DEFAULT_LENS_FILE,
    load_anchor_catalog,
    load_anchor_images,
)

DEFAULT_CKPT = (
    "/orcd/pool/007/pablomer/checkpoints_new/contrastive-spatial-conv1x1/"
    "best-epoch=53-step=83000.ckpt"
)
DEFAULT_OUT_DIR = "/orcd/pool/007/pablomer/outputs/contrastive_anomalies"


def load_contrastive(ckpt_path: str, device: torch.device):
    """Load a contrastive dual-encoder checkpoint (base or negative-ablation subclass)."""
    from galaxy_images.galaxy_model.contrastive_baseline.dual_encoder_contrastive import (
        DualEncoderContrastiveModule,
    )

    hp = torch.load(ckpt_path, map_location="cpu", weights_only=False).get("hyper_parameters", {})
    # The -DWNVAL arms carry three extra hparams and need the subclass; loading them
    # with the base class is the one way to get this silently wrong.
    if any(k in hp for k in ("galaxy_negatives", "instrument_negatives", "use_projection_head")):
        from galaxy_images.galaxy_model.contrastive_ablation.dual_encoder_contrastive_variants import (
            ContrastiveNegativeAblationModule as cls,
        )
    else:
        cls = DualEncoderContrastiveModule

    model = cls.load_from_checkpoint(ckpt_path, map_location="cpu")
    model.eval()
    torch.set_grad_enabled(False)
    if not (hasattr(model, "encoder_galaxy") and hasattr(model, "encoder_instrument")):
        raise ValueError(
            f"{ckpt_path} loaded as {type(model).__name__}, which has no "
            "encoder_galaxy/encoder_instrument pair — not a contrastive dual-encoder."
        )
    return model.to(device), dict(hp)


@torch.no_grad()
def _encode(model, images: np.ndarray, device: torch.device, batch_size: int, label: str):
    """Run both towers over a float16 (N,4,48,48) array. Returns (physics, instrument)."""
    n = len(images)
    phys, inst = [], []
    t0 = time.time()
    for i in range(0, n, batch_size):
        x = torch.from_numpy(images[i : i + batch_size].astype(np.float32)).to(device, non_blocking=True)
        phys.append(model.encoder_galaxy(x).flatten(1).float().cpu().numpy())
        inst.append(model.encoder_instrument(x).flatten(1).float().cpu().numpy())
        if i and i % (batch_size * 100) == 0:
            rate = i / max(time.time() - t0, 1e-6)
            print(f"[encode] {label}: {i:,}/{n:,}  ({rate:,.0f} img/s)", flush=True)
    print(f"[encode] {label}: done in {time.time() - t0:.1f}s", flush=True)
    return np.concatenate(phys).astype(np.float32), np.concatenate(inst).astype(np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=DEFAULT_CKPT)
    p.add_argument("--tag", default="spatial-conv1x1", help="Names the output file")
    p.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    p.add_argument("--lens-file", default=DEFAULT_LENS_FILE)
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--limit", type=int, default=None, help="Cap the anchor count (default: all)")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    device = torch.device(args.device)
    print(f"[encode] checkpoint : {args.ckpt}")
    print(f"[encode] device     : {device}")
    model, hp = load_contrastive(args.ckpt, device)
    d_phys = int(getattr(model.encoder_galaxy, "out_dim", hp.get("embedding_dim", 0)))
    d_inst = int(getattr(model.encoder_instrument, "out_dim", hp.get("embedding_dim", 0)))
    pool = hp.get("encoder_pool", "avg")
    print(f"[encode] pool={pool}  physics_dim={d_phys}  instrument_dim={d_inst}")
    if d_phys != d_inst:
        raise ValueError(f"tower widths differ ({d_phys} vs {d_inst}); the experiment assumes a match")

    cat = load_anchor_catalog(args.data_dir, args.lens_file, limit=args.limit)
    print(f"[encode] anchors: {len(cat):,}  (lenses: {int(cat.is_lens.sum())})")

    latents = {}
    for survey in ("hsc", "legacy"):
        imgs = load_anchor_images(args.data_dir, survey, cat.row_idx, cat.n_rows_total)
        ph, ins = _encode(model, imgs, device, args.batch_size, survey)
        latents[f"physics_{survey}"] = ph
        latents[f"instrument_{survey}"] = ins
        del imgs  # release ~1.9 GB before reading the next survey

    latents["physics_mean"] = (
        (latents["physics_hsc"] + latents["physics_legacy"]) / 2.0
    ).astype(np.float32)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"latents_{args.tag}.h5"
    fd, tmp = tempfile.mkstemp(suffix=".h5", prefix="latents_", dir=str(out_dir))
    os.close(fd)
    try:
        with h5py.File(tmp, "w") as f:
            f.create_dataset("row_idx", data=cat.row_idx, compression="gzip", compression_opts=4)
            f.create_dataset("ra", data=cat.ra, compression="gzip", compression_opts=4)
            f.create_dataset("dec", data=cat.dec, compression="gzip", compression_opts=4)
            f.create_dataset("is_lens", data=cat.is_lens, compression="gzip", compression_opts=4)
            str_dt = h5py.string_dtype("utf-8")
            f.create_dataset("object_id_hsc", data=cat.object_id_hsc.astype(object), dtype=str_dt)
            f.create_dataset("object_id_legacy", data=cat.object_id_legacy.astype(object), dtype=str_dt)
            for k, v in latents.items():
                f.create_dataset(k, data=v, compression="gzip", compression_opts=4)
                print(f"[encode] {k:20s} {v.shape}  mean={v.mean():+.3f} std={v.std():.3f}")
            f.attrs["checkpoint"] = args.ckpt
            f.attrs["tag"] = args.tag
            f.attrs["encoder_pool"] = str(pool)
            f.attrs["embedding_dim"] = d_phys
            f.attrs["n_anchors"] = len(cat)
            f.attrs["n_lenses"] = int(cat.is_lens.sum())
            f.attrs["data_dir"] = args.data_dir
            f.attrs["self_encoded"] = True
            f.attrs["model_class"] = type(model).__name__
        shutil.move(tmp, out_path)
    except Exception:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise

    print(f"[encode] saved: {out_path}")


if __name__ == "__main__":
    main()
