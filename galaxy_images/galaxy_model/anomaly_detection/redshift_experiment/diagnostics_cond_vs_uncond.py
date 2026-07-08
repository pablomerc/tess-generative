"""
Conditional-vs-unconditional NSF diagnostics (Fix 4).

Three independent tests, any one of which can diagnose "the cond flow is using
its context" vs "the cond flow ignored its context":

  D1: Score correlation between scores_uncond and scores_cond + top-32 overlap.
      Also RidgeCV + RandomForest R²(desi_z | latent) -- "could conditioning
      *possibly* help?".  -> outputs/D1_cond_vs_uncond.png + D3_metrics.txt

  D2: Context-sensitivity sweep. Retrain a fresh replica of the conditional
      flow, then hold a fixed latent x* and sweep z across the observed range;
      log p(x* | z) should vary non-trivially. Flat curves -> flow ignored
      context.  -> outputs/D2_z_sweep.png

Run from galaxy_model/:
  python anomaly_detection/redshift_experiment/diagnostics_cond_vs_uncond.py \
    [--profile wide|default] [--z-context z_z2|z] [--nsf-epochs 100] \
    [--webhook URL] [--device cuda]
"""
import argparse
from pathlib import Path

import h5py
import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import common  # noqa: E402

_HERE = Path(__file__).resolve().parent
OUTPUT_DIR = _HERE / "outputs"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--latents", type=Path, default=OUTPUT_DIR / "latents_redshift.h5")
    p.add_argument("--scores-uncond", type=Path, default=OUTPUT_DIR / "scores_uncond.npy")
    p.add_argument("--scores-cond", type=Path, default=OUTPUT_DIR / "scores_cond.npy")
    p.add_argument("--out-dir", type=Path, default=OUTPUT_DIR)
    p.add_argument("--top-n", type=int, default=32)
    p.add_argument("--profile", default="wide", choices=list(common.NSF_PROFILES.keys()))
    p.add_argument("--z-context", default="z_z2", choices=["z", "z_z2"])
    p.add_argument("--nsf-epochs", type=int, default=100)
    p.add_argument("--train-frac", type=float, default=0.8)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--n-fixed-latents", type=int, default=8,
                   help="Number of latents to hold fixed in the context-sensitivity sweep.")
    p.add_argument("--webhook", default=None)
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ----- load -----
    with h5py.File(args.latents, "r") as f:
        latents = f["hsc_mean"][:]
        z = f["desi_z"][:]
    su = np.load(args.scores_uncond)
    sc = np.load(args.scores_cond)
    n, d = latents.shape

    # ----- D1: correlations + R²(z|latent) -----
    from scipy.stats import spearmanr, pearsonr
    from sklearn.linear_model import RidgeCV
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split as sk_split
    from sklearn.metrics import r2_score

    sp = float(spearmanr(su, sc).statistic)
    pe = float(pearsonr(su, sc).statistic)
    ou, oc = np.argsort(-su)[: args.top_n], np.argsort(-sc)[: args.top_n]
    overlap = len(set(ou.tolist()) & set(oc.tolist()))
    sp_uz = float(spearmanr(su, z).statistic)
    sp_cz = float(spearmanr(sc, z).statistic)

    X_tr, X_te, y_tr, y_te = sk_split(latents, z, test_size=0.2, random_state=42)
    ridge = RidgeCV(alphas=np.logspace(-3, 3, 13)).fit(X_tr, y_tr)
    r2_ridge = float(r2_score(y_te, ridge.predict(X_te)))
    rf = RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=42).fit(X_tr, y_tr)
    r2_rf = float(r2_score(y_te, rf.predict(X_te)))

    diag_lines = [
        f"N = {n}, D = {d}",
        f"Spearman(uncond, cond)        = {sp:.4f}",
        f"Pearson (uncond, cond)        = {pe:.4f}",
        f"top-{args.top_n} overlap              = {overlap} / {args.top_n}",
        f"Spearman(uncond, z)           = {sp_uz:.4f}",
        f"Spearman(cond,   z)           = {sp_cz:.4f}",
        f"Ridge R²(z | latent)          = {r2_ridge:.4f}",
        f"RandomForest R²(z | latent)   = {r2_rf:.4f}",
    ]
    print("\n".join(diag_lines))

    (args.out_dir / "D3_metrics.txt").write_text("\n".join(diag_lines) + "\n")

    # D1 scatter
    fig, ax = plt.subplots(figsize=(7.5, 6))
    s = ax.scatter(su, sc, c=z, s=5, alpha=0.5, cmap="viridis")
    ax.scatter(su[ou], sc[ou], facecolors='none', edgecolors='red',    s=80,  lw=1.2, label=f"top-{args.top_n} uncond")
    ax.scatter(su[oc], sc[oc], facecolors='none', edgecolors='orange', s=120, lw=1.2, label=f"top-{args.top_n} cond")
    ax.set_xlabel("uncond NLL (Job A)")
    ax.set_ylabel("cond NLL p(latent|z) (Job B)")
    plt.colorbar(s, ax=ax, label="desi_z")
    ax.legend(loc="upper left")
    ax.set_title(
        f"cond-vs-uncond  Spearman {sp:.3f} | top-{args.top_n} overlap {overlap}/{args.top_n}\n"
        f"R²(z | latent) = {r2_ridge:.3f} (Ridge), {r2_rf:.3f} (RF)",
        fontsize=10)
    plt.tight_layout()
    d1_path = args.out_dir / "D1_cond_vs_uncond.png"
    plt.savefig(d1_path, dpi=140, bbox_inches="tight")
    plt.close()

    # ----- D2: retrained replica + z-sweep -----
    # Build z-context with the SAME mode and profile as Job B uses, so the
    # sweep reflects what Job B's flow actually learned (in expectation).
    train_idx, _ = common.train_test_split(n, args.train_frac)
    train_mask = np.zeros(n, dtype=bool); train_mask[train_idx] = True
    z_ctx = common.make_z_context(z, train_mask, mode=args.z_context)
    z_mu = float(z[train_idx].mean()); z_sd = float(z[train_idx].std() + 1e-8)

    print(f"\nRetraining replica conditional NSF (profile={args.profile}, "
          f"ctx={z_ctx.shape[1]}, epochs={args.nsf_epochs})...")
    _, flow = common.score_nsf(
        latents[train_idx], latents[:8], args.nsf_epochs, args.device,
        train_c=z_ctx[train_idx], all_c=z_ctx[:8],
        profile=args.profile, cosine_lr=True, return_flow=True)

    rng = np.random.default_rng(0)
    # Mix random latents with the top-anomaly latents (high-NLL) for variety.
    rand_idx = rng.choice(n, size=max(0, args.n_fixed_latents // 2), replace=False)
    top_idx = ou[: args.n_fixed_latents - len(rand_idx)]
    sweep_idx = np.concatenate([rand_idx, top_idx])
    fixed_lat = latents[sweep_idx]

    z_grid = np.linspace(float(z.min()), float(z.max()), 120)
    z_grid_std = ((z_grid - z_mu) / z_sd).astype(np.float32)
    if args.z_context == "z_z2":
        ctx_grid = np.stack([z_grid_std, z_grid_std ** 2], axis=1)
    else:
        ctx_grid = z_grid_std.reshape(-1, 1)

    # log p(x* | z) for each fixed latent across the z grid.
    device = torch.device(args.device)
    lat_t = torch.as_tensor(fixed_lat, dtype=torch.float32, device=device)
    logps = np.zeros((len(sweep_idx), len(z_grid)), dtype=np.float64)
    with torch.no_grad():
        for j, zg in enumerate(ctx_grid):
            zg_t = torch.as_tensor(zg, dtype=torch.float32, device=device).unsqueeze(0).expand(len(fixed_lat), -1)
            logps[:, j] = flow(zg_t).log_prob(lat_t).cpu().numpy()

    # Stdev of log p(x* | z) across z gives the "context sensitivity" per latent.
    sens = logps.std(axis=1)
    print(f"per-latent log p(x|z) stdev across z range:  {sens.round(3)}")
    print(f"median context-sensitivity = {np.median(sens):.3f}  (~0 ≡ context ignored)")

    fig, ax = plt.subplots(figsize=(9, 5))
    for k, idx in enumerate(sweep_idx):
        lab_kind = "random" if k < len(rand_idx) else "top-anom"
        ax.plot(z_grid, logps[k] - logps[k].mean(), label=f"{lab_kind} #{idx} (σ={sens[k]:.2f})",
                lw=1.2, alpha=0.85)
    ax.axhline(0, color="k", lw=0.5, alpha=0.3)
    ax.set_xlabel("DESI redshift z")
    ax.set_ylabel("log p(x* | z) − ⟨log p⟩  (centered per curve)")
    ax.set_title(f"D2 context-sensitivity sweep — replica cond flow "
                 f"(profile={args.profile}, ctx={args.z_context})\n"
                 f"flat curves ≡ flow ignored context; median σ = {np.median(sens):.3f}",
                 fontsize=10)
    ax.legend(fontsize=7, ncols=2, loc="best")
    plt.tight_layout()
    d2_path = args.out_dir / "D2_z_sweep.png"
    plt.savefig(d2_path, dpi=140, bbox_inches="tight")
    plt.close()

    # ----- Discord post -----
    verdict_overlap = ("cond ≈ uncond" if overlap >= int(0.7 * args.top_n) else
                       "cond ≠ uncond")
    verdict_sens = ("USING context" if np.median(sens) > 0.1 else
                    "IGNORING context")
    summary = (
        f"🔬 **Cond-NSF diagnostics** (Fix 4)\n"
        f"```\n" + "\n".join(diag_lines) + f"\n```\n"
        f"D2 median context-sensitivity σ(log p(x|z)) = **{np.median(sens):.3f}**  →  flow appears to be **{verdict_sens}**\n"
        f"top-{args.top_n} overlap (cond vs uncond) = {overlap}/{args.top_n}  →  {verdict_overlap}"
    )
    common.discord_notify(args.webhook, summary, file_path=d1_path)
    common.discord_notify(args.webhook, "D2: context-sensitivity sweep across z for 8 fixed latents.",
                          file_path=d2_path)

    print("Diagnostics done.")


if __name__ == "__main__":
    main()
