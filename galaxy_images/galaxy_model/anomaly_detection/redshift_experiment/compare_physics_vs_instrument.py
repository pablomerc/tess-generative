"""Head-to-head: physics-latent vs instrument-latent redshift-anomaly experiment.

Reads the two experiment output dirs (physics `outputs/`, instrument `outputs_instrument/`)
and quantifies the disentanglement contrast:
  - R2(z | latent) via ridge + random-forest (on the same 80/20 split, seed 42)
  - Spearman(uncond score, cond score) within each space (how much redshift-conditioning
    changes the anomaly ranking)
  - top-K anomaly overlap: within-space (uncond vs cond) and cross-space (physics vs instrument)

Both encode runs select the identical finite-desi_z>0 subset in catalog order, so rows are
aligned across the two H5 files (asserted). Produces a comparison figure + prints a summary;
optionally posts to Discord.

Usage:
  python compare_physics_vs_instrument.py [--physics-dir outputs] [--instr-dir outputs_instrument]
                                          [--top-k 32] [--webhook URL]
"""
import argparse
from pathlib import Path

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

import common

_HERE = Path(__file__).resolve().parent


def _r2(y, yhat):
    y = np.asarray(y, float); yhat = np.asarray(yhat, float)
    return float(1 - ((y - yhat) ** 2).sum() / (((y - y.mean()) ** 2).sum() + 1e-12))


def load_space(out_dir: Path, latents_name: str):
    with h5py.File(out_dir / latents_name, "r") as f:
        Z = f["hsc_mean"][:].astype(np.float32)
        z = f["desi_z"][:].astype(np.float64)
        rec = f["record_idx"][:].astype(np.int64)
    su = np.load(out_dir / "scores_uncond.npy")
    sc = np.load(out_dir / "scores_cond.npy")
    return dict(Z=Z, z=z, rec=rec, uncond=su, cond=sc)


def analyze(space, top_k):
    Z, z = space["Z"], space["z"]
    n = len(Z)
    tr, te = common.train_test_split(n, 0.8, seed=42)
    sc = StandardScaler().fit(Z[tr])
    Ztr, Zte = sc.transform(Z[tr]), sc.transform(Z[te])
    ridge = RidgeCV(alphas=np.logspace(-3, 4, 20)).fit(Ztr, z[tr])
    rf = RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=0).fit(Ztr, z[tr])
    r2_ridge = _r2(z[te], ridge.predict(Zte))
    r2_rf = _r2(z[te], rf.predict(Zte))
    rho = float(spearmanr(space["uncond"], space["cond"]).correlation)
    top_u = set(np.argsort(space["uncond"])[::-1][:top_k].tolist())
    top_c = set(np.argsort(space["cond"])[::-1][:top_k].tolist())
    return dict(r2_ridge=r2_ridge, r2_rf=r2_rf, spearman_uncond_cond=rho,
                overlap_uncond_cond=len(top_u & top_c), top_u=top_u, top_c=top_c)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--physics-dir", type=Path, default=_HERE / "outputs")
    p.add_argument("--instr-dir", type=Path, default=_HERE / "outputs_instrument")
    p.add_argument("--top-k", type=int, default=32)
    p.add_argument("--webhook", default=None)
    args = p.parse_args()

    phys = load_space(args.physics_dir, "latents_redshift.h5")
    inst = load_space(args.instr_dir, "latents_redshift_instrument.h5")
    assert np.array_equal(phys["rec"], inst["rec"]), "row misalignment between spaces!"

    P = analyze(phys, args.top_k)
    I = analyze(inst, args.top_k)
    xspace = len(P["top_u"] & I["top_u"])  # physics-top vs instrument-top (uncond)

    lines = [
        "PHYSICS vs INSTRUMENT — redshift anomaly experiment",
        f"N = {len(phys['Z'])} galaxies (aligned), latent dim {phys['Z'].shape[1]} / {inst['Z'].shape[1]}",
        "",
        f"{'metric':32s} {'physics':>10} {'instrument':>12}",
        f"{'R2(z | latent)  ridge':32s} {P['r2_ridge']:>10.3f} {I['r2_ridge']:>12.3f}",
        f"{'R2(z | latent)  random-forest':32s} {P['r2_rf']:>10.3f} {I['r2_rf']:>12.3f}",
        f"{'Spearman(uncond, cond)':32s} {P['spearman_uncond_cond']:>10.3f} {I['spearman_uncond_cond']:>12.3f}",
        f"{'top-{} overlap (uncond∩cond)'.format(args.top_k):32s} {P['overlap_uncond_cond']:>10d} {I['overlap_uncond_cond']:>12d}",
        "",
        f"cross-space top-{args.top_k} overlap (physics-uncond ∩ instrument-uncond): {xspace}/{args.top_k}",
        "",
        "Interpretation: if disentangled, instrument R2(z|latent) << physics, and instrument",
        "Spearman(uncond,cond) ~ 1 (redshift-conditioning barely changes instrument anomalies).",
    ]
    summary = "\n".join(lines)
    print(summary)
    (args.instr_dir / "compare_metrics.txt").write_text(summary + "\n")

    # figure: two panels
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6))
    x = np.arange(2); w = 0.35
    a1.bar(x - w / 2, [P["r2_ridge"], P["r2_rf"]], w, label="physics", color="#2c7fb8")
    a1.bar(x + w / 2, [I["r2_ridge"], I["r2_rf"]], w, label="instrument", color="#d95f02")
    a1.set_xticks(x); a1.set_xticklabels(["ridge", "random-forest"])
    a1.set_ylabel("R²(redshift | latent)"); a1.set_ylim(0, 1)
    a1.set_title("Redshift decodability by latent space\n(low instrument = disentangled)")
    a1.legend(); a1.grid(axis="y", alpha=0.3)

    a2.bar(x - w / 2, [P["spearman_uncond_cond"], P["overlap_uncond_cond"] / args.top_k], w,
           label="physics", color="#2c7fb8")
    a2.bar(x + w / 2, [I["spearman_uncond_cond"], I["overlap_uncond_cond"] / args.top_k], w,
           label="instrument", color="#d95f02")
    a2.set_xticks(x); a2.set_xticklabels(["Spearman(uncond,cond)", f"top-{args.top_k} overlap frac"])
    a2.set_ylabel("value"); a2.set_ylim(0, 1.05)
    a2.set_title("Effect of redshift-conditioning\n(high = conditioning barely matters)")
    a2.legend(); a2.grid(axis="y", alpha=0.3)
    fig.suptitle("Physics vs Instrument latent — redshift anomaly experiment", y=1.02)
    fig.tight_layout()
    out_png = args.instr_dir / "compare_physics_vs_instrument.png"
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    print("wrote", out_png)

    if args.webhook:
        common.discord_notify(args.webhook, "📊 Physics vs Instrument redshift-anomaly comparison:\n```\n"
                              + summary + "\n```", file_path=str(out_png))


if __name__ == "__main__":
    main()
