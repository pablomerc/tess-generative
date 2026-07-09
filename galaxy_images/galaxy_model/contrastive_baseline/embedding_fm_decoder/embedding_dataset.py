"""
Dataset for the embedding-conditioned FM decoder (Stage 2).

Loads the precomputed [embedding, image] H5 (see precompute_embeddings.py) fully into RAM
(like NeighborsPrecomputedDataset) and yields ``(cond_vec, target, survey_str)``.

The conditioning vector is always ``concat(emb_galaxy, emb_instrument)`` (dim = 2 * 64 = 128),
so the decoder architecture is fixed. Content ablations are done by ZEROING one half
(``cond_mode``), which keeps the conditioning dimension constant:

    concat128    -> [galaxy | instrument]        (default)
    galaxy64     -> [galaxy | 0]                 (probe: physics embedding only)
    instrument64 -> [0 | instrument]             (probe: instrument embedding only)
"""

from __future__ import annotations

import h5py
import torch
from torch.utils.data import Dataset, random_split


_COND_MODES = ("concat128", "galaxy64", "instrument64")


class EmbeddingImageDataset(Dataset):
    def __init__(self, h5_path: str, cond_mode: str = "concat128"):
        if cond_mode not in _COND_MODES:
            raise ValueError(f"cond_mode must be one of {_COND_MODES}, got {cond_mode!r}")
        self.cond_mode = cond_mode
        with h5py.File(h5_path, "r") as f:
            self.emb_galaxy = torch.from_numpy(f["emb_galaxy"][:]).float()
            self.emb_instrument = torch.from_numpy(f["emb_instrument"][:]).float()
            self.targets = torch.from_numpy(f["targets"][:]).float()
            self.survey = torch.from_numpy(f["survey"][:]).to(torch.int8)
        assert self.emb_galaxy.shape[0] == self.targets.shape[0]
        self.emb_dim = self.emb_galaxy.shape[1]
        self.cond_dim = 2 * self.emb_dim  # conditioning is always the concatenated vector
        print(f"[EmbeddingImageDataset] {len(self):,} examples | emb_dim={self.emb_dim} "
              f"| cond_dim={self.cond_dim} | cond_mode={cond_mode}")

    def __len__(self) -> int:
        return self.targets.shape[0]

    def __getitem__(self, idx: int):
        g = self.emb_galaxy[idx]
        ins = self.emb_instrument[idx]
        if self.cond_mode == "galaxy64":
            ins = torch.zeros_like(ins)
        elif self.cond_mode == "instrument64":
            g = torch.zeros_like(g)
        cond = torch.cat([g, ins], dim=0)  # (cond_dim,)
        survey = "hsc" if int(self.survey[idx]) == 0 else "legacy"
        return cond, self.targets[idx], survey


def embedding_collate(batch):
    """Collate to (cond_vecs, targets, metadata) — metadata mirrors the FM model's contract
    (list of dicts with 'anchor_survey') so per-survey loss/MSE splits work unchanged."""
    conds = torch.stack([b[0] for b in batch])
    targets = torch.stack([b[1] for b in batch])
    metadata = [{"anchor_survey": b[2]} for b in batch]
    return conds, targets, metadata


def make_train_val_split(dataset: Dataset, val_ratio: float = 0.08, seed: int = 42):
    n = len(dataset)
    n_val = int(n * val_ratio)
    n_train = n - n_val
    return random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(seed),
    )
