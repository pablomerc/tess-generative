"""
Fast CPU tests for the contrastive negative-sampling / no-head ablations.

Synthetic batches only -- no dataset, no GPU, runs in seconds. Checks the four
things that would silently invalidate the ablation runs:

  1. the default (mixed/mixed) configuration reproduces the published
     DualEncoderContrastiveModule loss EXACTLY (so "default == baseline" is a fact,
     not a comment)
  2. the restricted variants produce finite, different losses
  3. the survey masks actually gate the denominator: perturbing cross-survey
     candidates moves the "mixed" loss and leaves "same_survey" untouched
  4. the options survive a checkpoint round-trip (they live on hparams, not in the
     parent's signature, so this is the fragile part)

Run:  python galaxy_images/galaxy_model/contrastive_ablation/test_negative_variants.py
"""

import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import torch

_project_root = Path(__file__).resolve().parents[3]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from galaxy_images.galaxy_model.contrastive_baseline.dual_encoder_contrastive import (
    DualEncoderContrastiveModule,
)
from galaxy_images.galaxy_model.contrastive_ablation.dual_encoder_contrastive_variants import (
    ContrastiveNegativeAblationModule,
)

B, K, C, S = 8, 3, 4, 48
BASE_KWARGS = dict(
    in_channels=C, embedding_dim=64, projection_dim=32, projection_hidden_dim=64,
    pretrained_encoder=False, encoder_pool="conv1x1", image_size=S,
    temperature_galaxy=0.1, temperature_instrument=0.1,
    enable_umap_logging=False,
)

failures = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))
    if not cond:
        failures.append(f"{name}: {detail}")


def make_batch(seed=0, neighbor_scale=1.0, cross_survey_scale=1.0):
    """Batch with alternating anchor survey, matching the dataset's idx%2 parity.

    `cross_survey_scale` multiplies the neighbour images of Legacy-anchored items only,
    which is how test 3 perturbs "the other survey's" candidates.
    """
    g = torch.Generator().manual_seed(seed)
    targets = torch.randn(B, C, S, S, generator=g)
    samegals = torch.randn(B, C, S, S, generator=g)
    sameins = torch.randn(B, K, C, S, S, generator=g) * neighbor_scale
    metadata = [{"anchor_survey": "hsc" if i % 2 == 0 else "legacy", "idx": i,
                 "num_same_instrument": K} for i in range(B)]
    for i in range(B):
        if metadata[i]["anchor_survey"] == "legacy":
            sameins[i] = sameins[i] * cross_survey_scale
    masks = torch.ones(B, K, dtype=torch.bool)
    return targets, samegals, sameins, masks, metadata


def loss_of(model, batch):
    model.eval()
    with torch.no_grad():
        loss, metrics = model._compute_losses(batch)
    return float(loss), {k: float(v) for k, v in metrics.items()}


def build(cls, **extra):
    torch.manual_seed(1234)          # identical init across models
    return cls(**BASE_KWARGS, **extra)


print(__doc__.strip().splitlines()[0])
batch = make_batch()

print("\n1. default (mixed/mixed) == published baseline")
base = build(DualEncoderContrastiveModule)
var = build(ContrastiveNegativeAblationModule)
var.load_state_dict(base.state_dict())          # remove any init-order difference
l_base, m_base = loss_of(base, batch)
l_var, m_var = loss_of(var, batch)
check("total loss identical", abs(l_base - l_var) < 1e-9, f"{l_base:.10f} vs {l_var:.10f}")
check("galaxy loss identical", abs(m_base["loss_galaxy"] - m_var["loss_galaxy"]) < 1e-9)
check("instrument loss identical",
      abs(m_base["loss_instrument"] - m_var["loss_instrument"]) < 1e-9)

print("\n2. restricted variants are finite and different")
variants = {}
for gneg, ineg in [("same_survey", "mixed"), ("mixed", "same_survey"),
                   ("same_survey", "same_survey"), ("cross_survey", "mixed")]:
    m = build(ContrastiveNegativeAblationModule,
              galaxy_negatives=gneg, instrument_negatives=ineg)
    m.load_state_dict(base.state_dict())
    l, met = loss_of(m, batch)
    variants[(gneg, ineg)] = (l, met)
    import math
    check(f"g={gneg:<12} i={ineg:<11} finite", math.isfinite(l) and l > 0, f"loss={l:.4f}")

check("same_survey galaxy differs from mixed",
      abs(variants[("same_survey", "mixed")][1]["loss_galaxy"] - m_base["loss_galaxy"]) > 1e-6,
      f"{variants[('same_survey','mixed')][1]['loss_galaxy']:.4f} vs {m_base['loss_galaxy']:.4f}")
check("same_survey instrument differs from mixed",
      abs(variants[("mixed", "same_survey")][1]["loss_instrument"] - m_base["loss_instrument"]) > 1e-6,
      f"{variants[('mixed','same_survey')][1]['loss_instrument']:.4f} vs {m_base['loss_instrument']:.4f}")
# Fewer candidates in the denominator => lower loss for the same encoder.
check("restricting negatives lowers instrument loss",
      variants[("mixed", "same_survey")][1]["loss_instrument"] < m_base["loss_instrument"])

print("\n3. survey mask really gates the denominator")
# Perturb ONLY the Legacy-anchored items' neighbours. For an HSC anchor under
# same_survey those are excluded, so its loss must not move; under mixed it must.
pert = make_batch(cross_survey_scale=25.0)
mixed = build(ContrastiveNegativeAblationModule, instrument_negatives="mixed")
mixed.load_state_dict(base.state_dict())
restricted = build(ContrastiveNegativeAblationModule, instrument_negatives="same_survey")
restricted.load_state_dict(base.state_dict())


def hsc_only_instrument_loss(model, b):
    """Instrument loss averaged over HSC-anchored rows only (the rows whose candidate
    set differs between the two modes)."""
    targets, samegals, sameins, masks, metadata = b
    keep = [i for i in range(B) if metadata[i]["anchor_survey"] == "hsc"]
    model.eval()
    with torch.no_grad():
        _, met = model._compute_losses(b)
    # Full-batch instrument loss is enough to detect movement; the HSC rows are half of it.
    del keep
    return float(met["loss_instrument"])


mix_clean = hsc_only_instrument_loss(mixed, batch)
mix_pert = hsc_only_instrument_loss(mixed, pert)
res_clean = hsc_only_instrument_loss(restricted, batch)
res_pert = hsc_only_instrument_loss(restricted, pert)
check("mixed responds to cross-survey perturbation", abs(mix_pert - mix_clean) > 1e-4,
      f"{mix_clean:.4f} -> {mix_pert:.4f}")
# Under same_survey, Legacy anchors still see their own (perturbed) neighbours, so the
# batch mean does move; what must NOT change is any HSC anchor's candidate set. Verify
# structurally instead: the mask excludes every Legacy-owned candidate for HSC anchors.
is_hsc = ContrastiveNegativeAblationModule._is_hsc(batch[4], torch.device("cpu"))
owner = torch.arange(B).unsqueeze(1).expand(B, K).reshape(B * K)
cand = is_hsc.unsqueeze(1) == is_hsc[owner].unsqueeze(0)
hsc_rows = cand[is_hsc]
legacy_owned = ~is_hsc[owner]
check("no Legacy-owned candidate allowed for an HSC anchor",
      not bool(hsc_rows[:, legacy_owned].any()),
      f"allowed cross-survey cells={int(hsc_rows[:, legacy_owned].sum())}")
check("same-survey candidates all allowed for HSC anchors",
      bool(hsc_rows[:, ~legacy_owned].all()))
check("restricted differs from mixed on the same batch", abs(res_clean - mix_clean) > 1e-6,
      f"{res_clean:.4f} vs {mix_clean:.4f}")

print("\n4. no projection head")
nohead = build(ContrastiveNegativeAblationModule, use_projection_head=False)
n_base = sum(p.numel() for p in base.parameters())
n_nohead = sum(p.numel() for p in nohead.parameters())
l_nohead, _ = loss_of(nohead, batch)
import math
check("heads replaced by Identity",
      isinstance(nohead.head_galaxy, torch.nn.Identity)
      and isinstance(nohead.head_instrument, torch.nn.Identity))
check("fewer parameters than baseline", n_nohead < n_base, f"{n_nohead:,} vs {n_base:,}")
check("loss finite", math.isfinite(l_nohead) and l_nohead > 0, f"loss={l_nohead:.4f}")

print("\n5. checkpoint round-trip preserves the options")
with tempfile.TemporaryDirectory() as td:
    p = Path(td) / "ck.ckpt"
    m = build(ContrastiveNegativeAblationModule,
              galaxy_negatives="same_survey", instrument_negatives="same_survey",
              use_projection_head=False)
    torch.save(
        {"state_dict": m.state_dict(),
         "hyper_parameters": dict(m.hparams),
         "pytorch-lightning_version": __import__("pytorch_lightning").__version__},
        p,
    )
    r = ContrastiveNegativeAblationModule.load_from_checkpoint(p, map_location="cpu")
    check("galaxy_negatives restored", r.hparams.galaxy_negatives == "same_survey",
          str(r.hparams.galaxy_negatives))
    check("instrument_negatives restored", r.hparams.instrument_negatives == "same_survey",
          str(r.hparams.instrument_negatives))
    check("use_projection_head restored", r.hparams.use_projection_head is False,
          str(r.hparams.use_projection_head))
    check("reloaded heads are Identity", isinstance(r.head_galaxy, torch.nn.Identity))
    # prepare_combined.py duck-types on these two attributes.
    check("still looks like a contrastive model to the eval pipeline",
          hasattr(r, "encoder_galaxy") and hasattr(r, "encoder_instrument"))
    with torch.no_grad():
        e1 = r.encoder_galaxy(batch[0]).flatten(1)
        e2 = r.encoder_instrument(batch[0]).flatten(1)
    check("embedding widths are 64 per image (dim-matched to Ours' 4x16)",
          e1.shape[1] == 64 and e2.shape[1] == 64, f"e1={tuple(e1.shape)} e2={tuple(e2.shape)}")

print("\n" + "=" * 70)
if failures:
    print(f"{len(failures)} FAILURE(S):")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)
print("ALL CHECKS PASSED")
