"""
Counterfactual anomaly reconstruction experiment.

Three sub-experiments using a small set of hand-picked anomalies:
  - Physics examples: normal galaxies from Ours (Physics) top anomalies
  - Instrument examples: artifact galaxies from AION top anomalies

A — Style injection:
    samegal=HSC(phys), sameins=HSC(ins)×5  → does the artifact appear?

B — Normal reconstruction of instrument anomaly:
    samegal=Legacy(ins), sameins=HSC(phys)×k  → what does model think it looks like?

C — Artifact correction:
    samegal=HSC(ins), sameins=HSC(phys)×k  → can normal instrument context remove artifact?

Run from galaxy_model/:
  python anomaly_detection/counterfactual_anomaly_experiment/run_counterfactual.py
"""
import sys
from pathlib import Path

_here = Path(__file__).resolve().parent
_model_dir = _here.parent.parent
_repo_root = _model_dir.parent.parent
for p in [str(_model_dir), str(_repo_root)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import argparse
import numpy as np
import torch
# Force regular hipBLAS — hipBLASLt is buggy on MI210 for certain matrix shapes
torch.backends.cuda.preferred_blas_library("hipblas")
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from neighbors import preprocess_raw_image
from double_train_fm_neighbors import ConditionalFlowMatchingModule

# ── configuration ──────────────────────────────────────────────────────────────
PHYSICS_RANKS    = [459, 501, 434]
INSTRUMENT_RANKS = [2, 25, 46, 44, 76, 252, 269, 457]

NEIGHBORS_HDF5 = "/work1/jeroenaudenaert/pablomer/data/neighbours_v2.h5"
INDEX_NPZ      = _here / "outputs/top500_browse/rank_to_raw_index.npz"
CHECKPOINT     = _model_dir / "outputs/neighbors_all_attn/2026-04-05/checkpoints/best-epoch=228-step=87000.ckpt"
OUT_DIR        = _here / "outputs"

K_REPEAT   = 5    # how many times to repeat single instrument image in exp A
NUM_STEPS  = 100  # ODE integration steps (default 500; 100 is much faster)


# ── helpers ────────────────────────────────────────────────────────────────────
def _to_rgb(img_chw):
    """Percentile (1–99) stretch on first 3 channels → HxWx3 float [0,1]."""
    if isinstance(img_chw, torch.Tensor):
        img_chw = img_chw.detach().cpu().numpy()
    rgb = img_chw[:3].astype(np.float32)
    lo  = np.percentile(rgb, 1,  axis=(1, 2), keepdims=True)
    hi  = np.percentile(rgb, 99, axis=(1, 2), keepdims=True)
    return np.clip((rgb - lo) / (hi - lo + 1e-8), 0, 1).transpose(1, 2, 0)


def _load_hsc(f, raw_idx):
    img = torch.from_numpy(f["images_hsc"][raw_idx]).float()
    return preprocess_raw_image(img, "hsc", 48)[:4]          # (4,48,48)


def _load_legacy(f, raw_idx):
    img = torch.from_numpy(f["images_legacy"][raw_idx]).float()
    return preprocess_raw_image(img, "legacy", 48)[:4]        # (4,48,48)


def _sample(model, samegal_1chw, sameins_list, device, x_noise=None):
    """
    samegal_1chw : (4,48,48) tensor
    sameins_list : list of (4,48,48) tensors  (k images)
    Returns (4,48,48) reconstruction tensor on CPU.
    """
    k = len(sameins_list)
    sg  = samegal_1chw.unsqueeze(0).to(device)              # (1,4,48,48)
    si  = torch.stack(sameins_list).unsqueeze(0).to(device) # (1,k,4,48,48)
    msk = torch.ones(1, k, device=device)
    noise = x_noise.to(device) if x_noise is not None else None
    with torch.no_grad():
        out = model.sample(sg, si, masks=msk, x_noise=noise, num_steps=NUM_STEPS)
    return out[0].cpu()


# ── load resources ─────────────────────────────────────────────────────────────
def main(physics_ranks, instrument_ranks):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # rank → raw_index mapping
    idx = np.load(INDEX_NPZ)
    phys_raw = idx["physics_raw"]   # shape (512,)
    aion_raw = idx["aion_raw"]      # shape (512,)

    def phys_raw_idx(rank): return int(phys_raw[rank - 1])
    def aion_raw_idx(rank): return int(aion_raw[rank - 1])

    # load model
    print(f"Loading model from {CHECKPOINT} ...")
    model = ConditionalFlowMatchingModule.load_from_checkpoint(
        str(CHECKPOINT), map_location="cpu"
    )
    model.eval()
    torch.set_grad_enabled(False)
    model = model.to(device)
    print("Model loaded.")

    # pre-load all images
    print("Loading images from HDF5 ...")
    phys_hsc, phys_legacy, ins_hsc, ins_legacy = {}, {}, {}, {}
    with h5py.File(NEIGHBORS_HDF5, "r") as f:
        for r in physics_ranks:
            phys_hsc[r]    = _load_hsc(f, phys_raw_idx(r))
            phys_legacy[r] = _load_legacy(f, phys_raw_idx(r))
        for r in instrument_ranks:
            ins_hsc[r]    = _load_hsc(f, aion_raw_idx(r))
            ins_legacy[r] = _load_legacy(f, aion_raw_idx(r))
    print("Images loaded.")

    # fixed noise per physics example (for A) and per instrument example (for B, C)
    phys_noise = {r: torch.randn(1, 4, 48, 48) for r in physics_ranks}
    ins_noise  = {r: torch.randn(1, 4, 48, 48) for r in instrument_ranks}

    phys_hsc_list = [phys_hsc[r] for r in physics_ranks]
    k_phys = len(phys_hsc_list)
    img_size = 1.4
    n_phys = len(physics_ranks)
    n_ins  = len(instrument_ranks)

    # ── Sub-experiment A ───────────────────────────────────────────────────────
    print("\n=== Sub-experiment A: Style injection ===")

    recons_a = {}  # (phys_r, ins_r) -> (4,48,48) tensor
    n_cols_a = 1 + 2 * n_ins
    fig_a, axes_a = plt.subplots(n_phys, n_cols_a,
                                  figsize=(n_cols_a * img_size, n_phys * img_size + 0.6),
                                  squeeze=False)

    axes_a[0, 0].set_title("Physics\n(samegal)", fontsize=7, pad=2)
    for j, ins_r in enumerate(instrument_ranks):
        axes_a[0, 1 + 2*j].set_title(f"AION #{ins_r}\n(ins)", fontsize=7, pad=2)
        axes_a[0, 2 + 2*j].set_title(f"Recon", fontsize=7, pad=2)

    for i, phys_r in enumerate(physics_ranks):
        axes_a[i, 0].imshow(_to_rgb(phys_hsc[phys_r]))
        axes_a[i, 0].set_ylabel(f"Phys #{phys_r}", fontsize=7, labelpad=2)
        axes_a[i, 0].axis("off")

        for j, ins_r in enumerate(instrument_ranks):
            axes_a[i, 1 + 2*j].imshow(_to_rgb(ins_hsc[ins_r]))
            axes_a[i, 1 + 2*j].axis("off")

            sameins_list = [ins_hsc[ins_r]] * K_REPEAT
            recon = _sample(model, phys_hsc[phys_r], sameins_list, device,
                            x_noise=phys_noise[phys_r])
            recons_a[(phys_r, ins_r)] = recon
            axes_a[i, 2 + 2*j].imshow(_to_rgb(recon))
            axes_a[i, 2 + 2*j].axis("off")
            print(f"  A: phys#{phys_r} × ins#{ins_r} done")

    fig_a.suptitle("A — Style injection  (samegal=HSC(phys), sameins=HSC(ins)×5)",
                   fontsize=9, y=1.01)
    plt.tight_layout(pad=0.3)
    out_a = OUT_DIR / "exp_A_style_injection"
    out_a.mkdir(parents=True, exist_ok=True)
    fig_a.savefig(out_a / "style_injection_grid.png", dpi=180, bbox_inches="tight")
    plt.close(fig_a)

    # save tensors
    np.savez(out_a / "tensors.npz",
             physics_ranks=np.array(physics_ranks),
             instrument_ranks=np.array(instrument_ranks),
             phys_hsc=np.stack([phys_hsc[r].numpy() for r in physics_ranks]),
             ins_hsc=np.stack([ins_hsc[r].numpy() for r in instrument_ranks]),
             recons=np.array([[recons_a[(pr, ir)].numpy()
                               for ir in instrument_ranks]
                              for pr in physics_ranks]))
    print(f"Saved: {out_a}/style_injection_grid.png + tensors.npz")

    # ── Sub-experiment B ───────────────────────────────────────────────────────
    print("\n=== Sub-experiment B: Normal reconstruction of instrument anomalies ===")

    recons_b = {}  # ins_r -> (4,48,48) tensor
    # cols: Target (ins_hsc) | Galaxy pair (ins_legacy) | instr pair 1..k | Reconstruction
    n_cols_b = 1 + 1 + k_phys + 1
    fig_b, axes_b = plt.subplots(n_ins, n_cols_b,
                                  figsize=(n_cols_b * img_size, n_ins * img_size + 0.6),
                                  squeeze=False)

    axes_b[0, 0].set_title("Target\n(HSC)", fontsize=7, pad=2)
    axes_b[0, 1].set_title("Galaxy pair\n(Legacy)", fontsize=7, pad=2)
    for j, pr in enumerate(physics_ranks):
        axes_b[0, 2 + j].set_title(f"Instr. pair\nPhys#{pr}", fontsize=7, pad=2)
    axes_b[0, -1].set_title("Reconstruction", fontsize=7, pad=2)

    for i, ins_r in enumerate(instrument_ranks):
        axes_b[i, 0].imshow(_to_rgb(ins_hsc[ins_r]))
        axes_b[i, 0].set_ylabel(f"AION #{ins_r}", fontsize=7, labelpad=2)
        axes_b[i, 0].axis("off")
        axes_b[i, 1].imshow(_to_rgb(ins_legacy[ins_r]))
        axes_b[i, 1].axis("off")
        for j, pr in enumerate(physics_ranks):
            axes_b[i, 2 + j].imshow(_to_rgb(phys_hsc[pr]))
            axes_b[i, 2 + j].axis("off")

        recon = _sample(model, ins_legacy[ins_r], phys_hsc_list, device,
                        x_noise=ins_noise[ins_r])
        recons_b[ins_r] = recon
        axes_b[i, -1].imshow(_to_rgb(recon))
        axes_b[i, -1].axis("off")
        print(f"  B: ins#{ins_r} done")

    fig_b.suptitle("B — Normal reconstruction  (samegal=Legacy(ins), sameins=HSC(phys))",
                   fontsize=9, y=1.01)
    plt.tight_layout(pad=0.3)
    out_b = OUT_DIR / "exp_B_normal_recon"
    out_b.mkdir(parents=True, exist_ok=True)
    fig_b.savefig(out_b / "normal_recon.png", dpi=180, bbox_inches="tight")
    plt.close(fig_b)

    np.savez(out_b / "tensors.npz",
             physics_ranks=np.array(physics_ranks),
             instrument_ranks=np.array(instrument_ranks),
             ins_hsc=np.stack([ins_hsc[r].numpy() for r in instrument_ranks]),
             ins_legacy=np.stack([ins_legacy[r].numpy() for r in instrument_ranks]),
             phys_hsc=np.stack([phys_hsc[r].numpy() for r in physics_ranks]),
             recons=np.stack([recons_b[r].numpy() for r in instrument_ranks]))
    print(f"Saved: {out_b}/normal_recon.png + tensors.npz")

    # ── Sub-experiment C ───────────────────────────────────────────────────────
    print("\n=== Sub-experiment C: Artifact correction ===")

    recons_c = {}  # ins_r -> (4,48,48) tensor
    # cols: Target (ins_hsc) | instr pair 1..k | Reconstruction
    n_cols_c = 1 + k_phys + 1
    fig_c, axes_c = plt.subplots(n_ins, n_cols_c,
                                  figsize=(n_cols_c * img_size, n_ins * img_size + 0.6),
                                  squeeze=False)

    axes_c[0, 0].set_title("Target\n(HSC)", fontsize=7, pad=2)
    for j, pr in enumerate(physics_ranks):
        axes_c[0, 1 + j].set_title(f"Instr. pair\nPhys#{pr}", fontsize=7, pad=2)
    axes_c[0, -1].set_title("Reconstruction", fontsize=7, pad=2)

    for i, ins_r in enumerate(instrument_ranks):
        axes_c[i, 0].imshow(_to_rgb(ins_hsc[ins_r]))
        axes_c[i, 0].set_ylabel(f"AION #{ins_r}", fontsize=7, labelpad=2)
        axes_c[i, 0].axis("off")
        for j, pr in enumerate(physics_ranks):
            axes_c[i, 1 + j].imshow(_to_rgb(phys_hsc[pr]))
            axes_c[i, 1 + j].axis("off")

        recon = _sample(model, ins_hsc[ins_r], phys_hsc_list, device,
                        x_noise=ins_noise[ins_r])
        recons_c[ins_r] = recon
        axes_c[i, -1].imshow(_to_rgb(recon))
        axes_c[i, -1].axis("off")
        print(f"  C: ins#{ins_r} done")

    fig_c.suptitle("C — Artifact correction  (samegal=HSC(ins), sameins=HSC(phys))",
                   fontsize=9, y=1.01)
    plt.tight_layout(pad=0.3)
    out_c = OUT_DIR / "exp_C_artifact_correction"
    out_c.mkdir(parents=True, exist_ok=True)
    fig_c.savefig(out_c / "artifact_correction.png", dpi=180, bbox_inches="tight")
    plt.close(fig_c)

    np.savez(out_c / "tensors.npz",
             physics_ranks=np.array(physics_ranks),
             instrument_ranks=np.array(instrument_ranks),
             ins_hsc=np.stack([ins_hsc[r].numpy() for r in instrument_ranks]),
             phys_hsc=np.stack([phys_hsc[r].numpy() for r in physics_ranks]),
             recons=np.stack([recons_c[r].numpy() for r in instrument_ranks]))
    print(f"Saved: {out_c}/artifact_correction.png + tensors.npz")

    # ── Sub-experiment D ───────────────────────────────────────────────────────
    print("\n=== Sub-experiment D: Style injection with Legacy physics ===")

    recons_d = {}  # (phys_r, ins_r) -> (4,48,48) tensor
    n_cols_d = 1 + 2 * n_ins
    fig_d, axes_d = plt.subplots(n_phys, n_cols_d,
                                  figsize=(n_cols_d * img_size, n_phys * img_size + 0.6),
                                  squeeze=False)

    axes_d[0, 0].set_title("Physics\n(Legacy samegal)", fontsize=7, pad=2)
    for j, ins_r in enumerate(instrument_ranks):
        axes_d[0, 1 + 2*j].set_title(f"AION #{ins_r}\n(ins)", fontsize=7, pad=2)
        axes_d[0, 2 + 2*j].set_title(f"Recon", fontsize=7, pad=2)

    for i, phys_r in enumerate(physics_ranks):
        axes_d[i, 0].imshow(_to_rgb(phys_legacy[phys_r]))
        axes_d[i, 0].set_ylabel(f"Phys #{phys_r}", fontsize=7, labelpad=2)
        axes_d[i, 0].axis("off")

        for j, ins_r in enumerate(instrument_ranks):
            axes_d[i, 1 + 2*j].imshow(_to_rgb(ins_hsc[ins_r]))
            axes_d[i, 1 + 2*j].axis("off")

            sameins_list = [ins_hsc[ins_r]] * K_REPEAT
            recon = _sample(model, phys_legacy[phys_r], sameins_list, device,
                            x_noise=phys_noise[phys_r])
            recons_d[(phys_r, ins_r)] = recon
            axes_d[i, 2 + 2*j].imshow(_to_rgb(recon))
            axes_d[i, 2 + 2*j].axis("off")
            print(f"  D: phys#{phys_r} × ins#{ins_r} done")

    fig_d.suptitle("D — Style injection (Legacy)  (samegal=Legacy(phys), sameins=HSC(ins)×5)",
                   fontsize=9, y=1.01)
    plt.tight_layout(pad=0.3)
    out_d = OUT_DIR / "exp_D_style_injection_legacy"
    out_d.mkdir(parents=True, exist_ok=True)
    fig_d.savefig(out_d / "style_injection_legacy_grid.png", dpi=180, bbox_inches="tight")
    plt.close(fig_d)

    np.savez(out_d / "tensors.npz",
             physics_ranks=np.array(physics_ranks),
             instrument_ranks=np.array(instrument_ranks),
             phys_legacy=np.stack([phys_legacy[r].numpy() for r in physics_ranks]),
             ins_hsc=np.stack([ins_hsc[r].numpy() for r in instrument_ranks]),
             recons=np.array([[recons_d[(pr, ir)].numpy()
                               for ir in instrument_ranks]
                              for pr in physics_ranks]))
    print(f"Saved: {out_d}/style_injection_legacy_grid.png + tensors.npz")

    # ── Sub-experiment E ───────────────────────────────────────────────────────
    print("\n=== Sub-experiment E: Artifact correction with random HSC pairs ===")

    # Draw k_phys random HSC images from the middle of the dataset (avoid top anomalies)
    rng = np.random.default_rng(42)
    n_total_hsc = 468197
    rand_raw_idxs = rng.integers(50000, n_total_hsc - 50000, size=k_phys).tolist()
    rand_hsc = {}
    with h5py.File(NEIGHBORS_HDF5, "r") as f:
        for raw_idx in rand_raw_idxs:
            rand_hsc[raw_idx] = _load_hsc(f, raw_idx)
    rand_hsc_list = [rand_hsc[i] for i in rand_raw_idxs]
    print(f"  Random HSC raw indices: {rand_raw_idxs}")

    recons_e = {}  # ins_r -> (4,48,48) tensor
    n_cols_e = 1 + k_phys + 1
    fig_e, axes_e = plt.subplots(n_ins, n_cols_e,
                                  figsize=(n_cols_e * img_size, n_ins * img_size + 0.6),
                                  squeeze=False)

    axes_e[0, 0].set_title("Target\n(HSC)", fontsize=7, pad=2)
    for j, raw_idx in enumerate(rand_raw_idxs):
        axes_e[0, 1 + j].set_title(f"Rand HSC\n#{raw_idx}", fontsize=7, pad=2)
    axes_e[0, -1].set_title("Reconstruction", fontsize=7, pad=2)

    for i, ins_r in enumerate(instrument_ranks):
        axes_e[i, 0].imshow(_to_rgb(ins_hsc[ins_r]))
        axes_e[i, 0].set_ylabel(f"AION #{ins_r}", fontsize=7, labelpad=2)
        axes_e[i, 0].axis("off")
        for j, raw_idx in enumerate(rand_raw_idxs):
            axes_e[i, 1 + j].imshow(_to_rgb(rand_hsc[raw_idx]))
            axes_e[i, 1 + j].axis("off")

        recon = _sample(model, ins_hsc[ins_r], rand_hsc_list, device,
                        x_noise=ins_noise[ins_r])
        recons_e[ins_r] = recon
        axes_e[i, -1].imshow(_to_rgb(recon))
        axes_e[i, -1].axis("off")
        print(f"  E: ins#{ins_r} done")

    fig_e.suptitle("E — Artifact correction (random HSC)  (samegal=HSC(ins), sameins=random HSC)",
                   fontsize=9, y=1.01)
    plt.tight_layout(pad=0.3)
    out_e = OUT_DIR / "exp_E_artifact_correction_random"
    out_e.mkdir(parents=True, exist_ok=True)
    fig_e.savefig(out_e / "artifact_correction_random.png", dpi=180, bbox_inches="tight")
    plt.close(fig_e)

    np.savez(out_e / "tensors.npz",
             instrument_ranks=np.array(instrument_ranks),
             rand_raw_idxs=np.array(rand_raw_idxs),
             ins_hsc=np.stack([ins_hsc[r].numpy() for r in instrument_ranks]),
             rand_hsc=np.stack([rand_hsc[i].numpy() for i in rand_raw_idxs]),
             recons=np.stack([recons_e[r].numpy() for r in instrument_ranks]))
    print(f"Saved: {out_e}/artifact_correction_random.png + tensors.npz")

    print("\nAll done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--physics-ranks", type=int, nargs="+", default=PHYSICS_RANKS)
    parser.add_argument("--instrument-ranks", type=int, nargs="+", default=INSTRUMENT_RANKS)
    args = parser.parse_args()
    main(args.physics_ranks, args.instrument_ranks)
