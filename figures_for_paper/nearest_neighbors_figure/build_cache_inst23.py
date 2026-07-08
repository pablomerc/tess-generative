"""
build_cache_inst23.py — variant of build_cache.py with instrument NN ranks [2, 3]
instead of [1, 2]. Writes _cache/query_172_inst23.npz.
"""
import sys
from pathlib import Path

_repo = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_repo))
sys.path.insert(0, str(_repo / "galaxy_images" / "galaxy_model"))

import h5py
import numpy as np
from sklearn.neighbors import NearestNeighbors
from neighbors import NeighborsSimpleDataset

LATENTS      = _repo / "galaxy_images/galaxy_model/neighbor_search/neighbor_latents_103k.h5"
NEIGHBORS_H5 = Path("/work1/jeroenaudenaert/pablomer/data/neighbours_v2.h5")
OUT          = Path(__file__).resolve().parent / "_cache" / "query_172_inst23.npz"

QUERY_IDX      = 172
HSC_PHYS_RANKS = [1, 2, 3, 4, 6]
LEG_PHYS_RANKS = [1, 2, 3, 5, 9]
INST_RANKS     = [2, 3]
K_FETCH        = 15


def main():
    print("Loading embeddings...")
    with h5py.File(LATENTS, "r") as f:
        hsc_phys = f["physics_embedding"][:]
        hsc_inst = f["instrument_embedding"][:]
        leg_phys = f["legacy_physics_embedding"][:]
        leg_inst = f["legacy_instrument_embedding"][:]

    n = len(hsc_phys)
    comb_phys = np.concatenate([hsc_phys, leg_phys], axis=0)
    comb_inst = np.concatenate([hsc_inst, leg_inst], axis=0)

    print("Fitting kNN...")
    nn_phys = NearestNeighbors(n_neighbors=K_FETCH+2, metric="euclidean").fit(comb_phys)
    nn_inst = NearestNeighbors(n_neighbors=K_FETCH+2, metric="euclidean").fit(comb_inst)

    def get_neighbors(query_emb, nn_model, self_pos, ranks):
        _, inds = nn_model.kneighbors(query_emb, n_neighbors=K_FETCH+2)
        valid = [p for p in inds[0] if p != self_pos]
        return [(int(valid[r-1] % n), "hsc" if valid[r-1] < n else "legacy", r)
                for r in ranks if r-1 < len(valid)]

    hsc_phys_nbs = get_neighbors(comb_phys[QUERY_IDX:QUERY_IDX+1],     nn_phys, QUERY_IDX,   HSC_PHYS_RANKS)
    hsc_inst_nbs = get_neighbors(comb_inst[QUERY_IDX:QUERY_IDX+1],     nn_inst, QUERY_IDX,   INST_RANKS)
    leg_phys_nbs = get_neighbors(comb_phys[n+QUERY_IDX:n+QUERY_IDX+1], nn_phys, n+QUERY_IDX, LEG_PHYS_RANKS)
    leg_inst_nbs = get_neighbors(comb_inst[n+QUERY_IDX:n+QUERY_IDX+1], nn_inst, n+QUERY_IDX, INST_RANKS)

    print("Loading images via NeighborsSimpleDataset...")
    ds = NeighborsSimpleDataset(hdf5_path=str(NEIGHBORS_H5))

    def load(idx):
        h, l, _ = ds[idx]
        return h.numpy(), l.numpy()

    def fetch(nbs):
        imgs, srcs, ranks = [], [], []
        for didx, src, rank in nbs:
            h, l = load(didx)
            imgs.append(h if src == "hsc" else l)
            srcs.append(src)
            ranks.append(rank)
        return np.array(imgs), np.array(srcs), np.array(ranks)

    q_hsc, q_leg = load(QUERY_IDX)
    hsc_pi, hsc_ps, hsc_pr = fetch(hsc_phys_nbs)
    hsc_ii, hsc_is, hsc_ir = fetch(hsc_inst_nbs)
    leg_pi, leg_ps, leg_pr = fetch(leg_phys_nbs)
    leg_ii, leg_is, leg_ir = fetch(leg_inst_nbs)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUT,
        query_hsc=q_hsc, query_legacy=q_leg,
        hsc_phys_imgs=hsc_pi, hsc_phys_srcs=hsc_ps, hsc_phys_ranks=hsc_pr,
        hsc_inst_imgs=hsc_ii, hsc_inst_srcs=hsc_is, hsc_inst_ranks=hsc_ir,
        leg_phys_imgs=leg_pi, leg_phys_srcs=leg_ps, leg_phys_ranks=leg_pr,
        leg_inst_imgs=leg_ii, leg_inst_srcs=leg_is, leg_inst_ranks=leg_ir,
    )
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
