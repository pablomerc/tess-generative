"""
Cross-space PCA + t-SNE visualization of anomaly detectors.

Mirrors plot_umap_cross_space.py but uses PCA and t-SNE.
For PCA:   fit on background, transform anomalies separately.
For t-SNE: no out-of-sample transform — embed background + anomalies together,
           then split. High-D spaces are PCA-pre-reduced to 50D first.
"""
import argparse
import functools
import time
from pathlib import Path

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_here = Path(__file__).resolve().parent
OUTPUT_DIR = _here.parent / "outputs"


def _resolve_path(path_str):
    p = Path(path_str)
    if not p.is_absolute() and not p.exists():
        p = OUTPUT_DIR / path_str
    return p


def _top_n_by_score(scores_path, score_key, top_n):
    with h5py.File(scores_path, "r") as f:
        scores = f[score_key][:]
    return np.argsort(scores)[::-1][:top_n]


def _fit_pca_transform(bg_feats, query_feats, include_anomalies=False):
    from sklearn.decomposition import PCA
    t0 = time.time()
    pca = PCA(n_components=2, random_state=42)
    if include_anomalies:
        n_bg = len(bg_feats)
        combined = np.vstack([bg_feats, query_feats]).astype(np.float32)
        emb = pca.fit_transform(combined)
        bg_emb, query_emb = emb[:n_bg], emb[n_bg:]
    else:
        bg_emb = pca.fit_transform(bg_feats.astype(np.float32))
        query_emb = pca.transform(query_feats.astype(np.float32))
    print(f"    done in {time.time()-t0:.1f}s  "
          f"(var explained: {pca.explained_variance_ratio_.sum():.3f})")
    return bg_emb, query_emb


def _fit_tsne_transform(bg_feats, query_feats, n_jobs=1):
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    n_bg = len(bg_feats)
    combined = np.vstack([bg_feats, query_feats]).astype(np.float32)

    if combined.shape[1] > 100:
        print(f"    PCA pre-reduction: {combined.shape[1]}D → 50D ...")
        combined = PCA(n_components=50, random_state=42).fit_transform(combined)

    print(f"    t-SNE on {combined.shape[0]} pts × {combined.shape[1]}D "
          f"(n_jobs={n_jobs}) ...")
    t0 = time.time()
    tsne = TSNE(n_components=2, perplexity=30, max_iter=1000,
                random_state=42, verbose=1, n_jobs=n_jobs)
    emb = tsne.fit_transform(combined)
    print(f"    done in {time.time()-t0:.1f}s")
    return emb[:n_bg], emb[n_bg:]


def _make_figure(bg_embs, space_names, anom_embs_per_set, highlight_defs, out_path, title):
    n = len(space_names)
    fig, axes = plt.subplots(1, n, figsize=(n * 5, 5))
    if n == 1:
        axes = [axes]

    for col, (ax, name, bg) in enumerate(zip(axes, space_names, bg_embs)):
        ax.scatter(bg[:, 0], bg[:, 1], c="#cccccc", s=2, alpha=0.4,
                   rasterized=True, linewidths=0)
        for set_name, (color, marker, label) in highlight_defs.items():
            emb = anom_embs_per_set[set_name][col]
            ax.scatter(emb[:, 0], emb[:, 1], c=color, s=140, marker=marker,
                       label=label, zorder=5, edgecolors="black", linewidths=0.6)
        ax.set_title(name, fontsize=11)
        ax.axis("off")
        if col == 0:
            ax.legend(fontsize=8, loc="upper left", framealpha=0.8)

    fig.suptitle(title, fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def _run_method(name, fit_fn, space_configs, bg_pos, anom_sorted,
                phys_top, aion_top, ins_top, out_dir, suffix, top_n, pos_to_row):

    def _rows(pos_arr):
        return np.array([pos_to_row[int(p)] for p in pos_arr])

    bg_embs = []
    anom_embs = {k: [] for k in ("physics", "aion", "instrument")}

    for space_name, feats in space_configs:
        print(f"\n=== {name}: {space_name} ===")
        bg_emb, query_emb = fit_fn(feats[bg_pos], feats[anom_sorted])
        bg_embs.append(bg_emb)
        anom_embs["physics"].append(query_emb[_rows(phys_top)])
        anom_embs["aion"].append(query_emb[_rows(aion_top)])
        anom_embs["instrument"].append(query_emb[_rows(ins_top)])

    space_names = [cfg[0] for cfg in space_configs]
    tag = name.lower().replace("-", "").replace(" ", "")

    _make_figure(
        bg_embs, space_names,
        {k: anom_embs[k] for k in ("physics", "aion")},
        {
            "physics": ("#e74c3c", "o", f"Physics top-{top_n} (iforest)"),
            "aion":    ("#3498db", "^", f"AION top-{top_n} (iforest)"),
        },
        out_dir / f"{tag}_crossspace_physics_aion_{suffix}.png",
        f"Cross-space {name} — physics & AION anomalies",
    )
    _make_figure(
        bg_embs, space_names,
        {"instrument": anom_embs["instrument"]},
        {"instrument": ("#2ecc71", "*", f"Instrument top-{top_n} (iforest)")},
        out_dir / f"{tag}_crossspace_instrument_{suffix}.png",
        f"Cross-space {name} — instrument anomalies",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ours-latents", default="anomaly_latents_ours_ours_100k.h5")
    parser.add_argument("--aion-latents", default="anomaly_latents_aion_aion_100k.h5")
    parser.add_argument("--ins-latents",  default="anomaly_latents_ours_ins_100k.h5")
    parser.add_argument("--ours-scores",  default="anomaly_scores_ours_100k.h5")
    parser.add_argument("--aion-scores",  default="anomaly_scores_aion_100k.h5")
    parser.add_argument("--ins-scores",   default="anomaly_scores_ins_100k.h5")
    parser.add_argument("--suffix",            default="crossspace_100k")
    parser.add_argument("--n-background",      type=int, default=10000)
    parser.add_argument("--top-n",             type=int, default=30)
    parser.add_argument("--n-jobs",            type=int, default=1,
                        help="Threads for t-SNE nearest-neighbour search")
    parser.add_argument("--include-anomalies", action="store_true",
                        help="Include anomaly points in PCA fit (t-SNE always includes them)")
    args = parser.parse_args()

    out_dir = OUTPUT_DIR / f"figures_{args.suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading latents...")
    with h5py.File(_resolve_path(args.ours_latents), "r") as f:
        physics_feats = f["hsc_flat"][:]
    with h5py.File(_resolve_path(args.aion_latents), "r") as f:
        aion_feats = f["embeddings_mean_hsc"][:]
    with h5py.File(_resolve_path(args.ins_latents), "r") as f:
        ins_feats = f["hsc_flat"][:]

    n_total = len(physics_feats)
    print(f"  N={n_total}  physics={physics_feats.shape}  "
          f"aion={aion_feats.shape}  ins={ins_feats.shape}")

    print(f"\nTop-{args.top_n} iforest anomalies per detector...")
    phys_top = _top_n_by_score(_resolve_path(args.ours_scores), "ours/hsc_flat/iforest", args.top_n)
    aion_top = _top_n_by_score(_resolve_path(args.aion_scores), "aion/hsc_mean_pca64/iforest", args.top_n)
    ins_top  = _top_n_by_score(_resolve_path(args.ins_scores),  "ours/hsc_flat/iforest", args.top_n)

    all_anom_pos = np.unique(np.concatenate([phys_top, aion_top, ins_top]))
    anom_set = set(all_anom_pos.tolist())
    candidates = np.array([i for i in range(n_total) if i not in anom_set])
    rng = np.random.default_rng(42)
    bg_pos = np.sort(rng.choice(candidates, size=min(args.n_background, len(candidates)), replace=False))
    print(f"Background: {len(bg_pos)} points (excluded {len(anom_set)} anomaly positions)")

    anom_sorted = np.sort(all_anom_pos)
    pos_to_row = {int(p): r for r, p in enumerate(anom_sorted)}

    space_configs = [
        ("Physics (64D)",    physics_feats),
        ("AION (768D)",      aion_feats),
        ("Instrument (64D)", ins_feats),
    ]

    tsne_fn = functools.partial(_fit_tsne_transform, n_jobs=args.n_jobs)
    pca_fn  = functools.partial(_fit_pca_transform, include_anomalies=args.include_anomalies)

    _run_method("PCA",   pca_fn,  space_configs, bg_pos, anom_sorted,
                phys_top, aion_top, ins_top, out_dir, args.suffix, args.top_n, pos_to_row)
    _run_method("t-SNE", tsne_fn, space_configs, bg_pos, anom_sorted,
                phys_top, aion_top, ins_top, out_dir, args.suffix, args.top_n, pos_to_row)

    print("\nDone.")


if __name__ == "__main__":
    main()
