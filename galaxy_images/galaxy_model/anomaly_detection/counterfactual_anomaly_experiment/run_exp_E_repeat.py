"""
Repeat of sub-experiment E (artifact correction with random HSC pairs).

Differences vs. the original exp_E:
  - 10 instrument anomalies = original 8 + AION #28, #337
    (AION #25 and #76 already in original list, so adding only the new two)
  - 10 random HSC images as instrument-encoder inputs
  - random HSC pairs are filtered to skip empty/zero entries in the HDF5
    (some raw_indices in neighbours_v2.h5 contain all-zero stamps; those
    rendered as black tiles in the previous run)
  - Each row shows two reconstructions:
      * single  — flow-matched recon from one fixed noise sample
      * mean    — pixel-mean of 5 recons drawn from independent noise

Run from galaxy_model/:
  python anomaly_detection/counterfactual_anomaly_experiment/run_exp_E_repeat.py
"""
import sys
from pathlib import Path

_here = Path(__file__).resolve().parent
_model_dir = _here.parent.parent
_repo_root = _model_dir.parent.parent
for p in [str(_model_dir), str(_repo_root)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import torch
torch.backends.cuda.preferred_blas_library("hipblas")
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from neighbors import preprocess_raw_image
from double_train_fm_neighbors import ConditionalFlowMatchingModule

# ── configuration ──────────────────────────────────────────────────────────────
# original 8 instrument ranks, plus the 2 new ones from /preprocessed (28, 337);
# 25 and 76 from the user's request are already in the original list.
INSTRUMENT_RANKS = [2, 25, 28, 44, 46, 76, 252, 269, 337, 457]
N_PAIRS = 10
N_POSTERIOR = 5

NEIGHBORS_HDF5 = "/work1/jeroenaudenaert/pablomer/data/neighbours_v2.h5"
INDEX_NPZ      = _here / "outputs/top500_browse/rank_to_raw_index.npz"
CHECKPOINT     = _model_dir / "outputs/neighbors_all_attn/2026-04-05/checkpoints/best-epoch=228-step=87000.ckpt"
OUT_DIR        = _here / "outputs" / "exp_E_repeat_random_10pairs"

NUM_STEPS = 100
SEED      = 42


# ── helpers ────────────────────────────────────────────────────────────────────
def _to_rgb(img_chw):
    if isinstance(img_chw, torch.Tensor):
        img_chw = img_chw.detach().cpu().numpy()
    rgb = img_chw[:3].astype(np.float32)
    lo  = np.percentile(rgb, 1,  axis=(1, 2), keepdims=True)
    hi  = np.percentile(rgb, 99, axis=(1, 2), keepdims=True)
    return np.clip((rgb - lo) / (hi - lo + 1e-8), 0, 1).transpose(1, 2, 0)


def _load_hsc(f, raw_idx):
    img = torch.from_numpy(f["images_hsc"][raw_idx]).float()
    return preprocess_raw_image(img, "hsc", 48)[:4]  # (4,48,48)


def _is_valid_raw(f, raw_idx):
    """A raw HSC entry is invalid if it is all-zero in the HDF5."""
    img = f["images_hsc"][raw_idx]
    return float(np.abs(img).max()) > 0.0


def _draw_valid_random_indices(f, n, rng, low, high):
    """Draw n random raw HSC indices in [low, high) that are non-empty."""
    chosen = []
    seen = set()
    while len(chosen) < n:
        cand = int(rng.integers(low, high))
        if cand in seen:
            continue
        seen.add(cand)
        if _is_valid_raw(f, cand):
            chosen.append(cand)
    return chosen


def _sample(model, samegal_1chw, sameins_list, device, x_noise):
    k = len(sameins_list)
    sg  = samegal_1chw.unsqueeze(0).to(device)
    si  = torch.stack(sameins_list).unsqueeze(0).to(device)
    msk = torch.ones(1, k, device=device)
    with torch.no_grad():
        out = model.sample(sg, si, masks=msk,
                           x_noise=x_noise.to(device),
                           num_steps=NUM_STEPS)
    return out[0].cpu()


# ── main ───────────────────────────────────────────────────────────────────────
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    idx = np.load(INDEX_NPZ)
    aion_raw = idx["aion_raw"]

    def aion_raw_idx(rank):
        return int(aion_raw[rank - 1])

    print(f"Loading model from {CHECKPOINT} ...")
    model = ConditionalFlowMatchingModule.load_from_checkpoint(
        str(CHECKPOINT), map_location="cpu"
    )
    model.eval()
    torch.set_grad_enabled(False)
    model = model.to(device)
    print("Model loaded.")

    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)

    print("Loading instrument anomalies + drawing 10 valid random HSC pairs ...")
    ins_hsc = {}
    with h5py.File(NEIGHBORS_HDF5, "r") as f:
        n_total_hsc = f["images_hsc"].shape[0]
        for r in INSTRUMENT_RANKS:
            ins_hsc[r] = _load_hsc(f, aion_raw_idx(r))

        rand_raw_idxs = _draw_valid_random_indices(
            f, N_PAIRS, rng,
            low=50_000, high=n_total_hsc - 50_000
        )
        print(f"  Random HSC raw indices (filtered): {rand_raw_idxs}")
        rand_hsc = {i: _load_hsc(f, i) for i in rand_raw_idxs}
    rand_hsc_list = [rand_hsc[i] for i in rand_raw_idxs]

    # 5 noise tensors per instrument anomaly (reused across single/mean)
    ins_noises = {
        r: torch.stack([torch.randn(4, 48, 48) for _ in range(N_POSTERIOR)])
        for r in INSTRUMENT_RANKS
    }

    n_ins = len(INSTRUMENT_RANKS)
    img_size = 1.4
    # cols: Target | 10 random HSC pairs | Single recon | Mean recon
    n_cols = 1 + N_PAIRS + 2

    fig, axes = plt.subplots(
        n_ins, n_cols,
        figsize=(n_cols * img_size, n_ins * img_size + 0.6),
        squeeze=False,
    )

    axes[0, 0].set_title("Target\n(HSC)", fontsize=7, pad=2)
    for j, raw_idx in enumerate(rand_raw_idxs):
        axes[0, 1 + j].set_title(f"Rand HSC\n#{raw_idx}", fontsize=7, pad=2)
    axes[0, 1 + N_PAIRS].set_title("Recon\n(single)", fontsize=7, pad=2)
    axes[0, 2 + N_PAIRS].set_title(f"Recon\n(mean of {N_POSTERIOR})", fontsize=7, pad=2)

    recons_single = {}
    recons_mean   = {}
    recons_all    = {}  # ins_r -> (N_POSTERIOR, 4, 48, 48)

    for i, ins_r in enumerate(INSTRUMENT_RANKS):
        axes[i, 0].imshow(_to_rgb(ins_hsc[ins_r]))
        axes[i, 0].set_ylabel(f"AION #{ins_r}", fontsize=7, labelpad=2)
        axes[i, 0].axis("off")

        for j, raw_idx in enumerate(rand_raw_idxs):
            axes[i, 1 + j].imshow(_to_rgb(rand_hsc[raw_idx]))
            axes[i, 1 + j].axis("off")

        # Draw N_POSTERIOR independent posterior samples
        samples = []
        for s in range(N_POSTERIOR):
            x_noise = ins_noises[ins_r][s].unsqueeze(0)  # (1,4,48,48)
            recon = _sample(model, ins_hsc[ins_r], rand_hsc_list, device, x_noise)
            samples.append(recon)
        samples_t = torch.stack(samples)  # (N_POSTERIOR, 4, 48, 48)
        recons_all[ins_r]    = samples_t
        recons_single[ins_r] = samples[0]
        recons_mean[ins_r]   = samples_t.mean(dim=0)

        axes[i, 1 + N_PAIRS].imshow(_to_rgb(recons_single[ins_r]))
        axes[i, 1 + N_PAIRS].axis("off")
        axes[i, 2 + N_PAIRS].imshow(_to_rgb(recons_mean[ins_r]))
        axes[i, 2 + N_PAIRS].axis("off")
        print(f"  E*: ins#{ins_r} done ({N_POSTERIOR} posterior samples)")

    fig.suptitle(
        "E (repeat) — Artifact correction (random HSC, 10 pairs)\n"
        "samegal=HSC(ins), sameins=10×random HSC; right two cols: single recon vs. mean of 5",
        fontsize=9, y=1.01,
    )
    plt.tight_layout(pad=0.3)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_png = OUT_DIR / "artifact_correction_random_10pairs.png"
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {out_png}")

    np.savez(
        OUT_DIR / "tensors.npz",
        instrument_ranks=np.array(INSTRUMENT_RANKS),
        rand_raw_idxs=np.array(rand_raw_idxs),
        ins_hsc=np.stack([ins_hsc[r].numpy() for r in INSTRUMENT_RANKS]),
        rand_hsc=np.stack([rand_hsc[i].numpy() for i in rand_raw_idxs]),
        recons_single=np.stack([recons_single[r].numpy() for r in INSTRUMENT_RANKS]),
        recons_mean=np.stack([recons_mean[r].numpy() for r in INSTRUMENT_RANKS]),
        recons_all=np.stack([recons_all[r].numpy() for r in INSTRUMENT_RANKS]),
    )
    print(f"Saved tensors: {OUT_DIR / 'tensors.npz'}")


if __name__ == "__main__":
    main()
