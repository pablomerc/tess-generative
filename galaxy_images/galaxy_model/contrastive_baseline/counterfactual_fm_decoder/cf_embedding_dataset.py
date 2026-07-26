"""
Dataset for the counterfactual embedding-conditioned FM decoder (Stage 2).

Loads the precomputed CF H5 (see precompute_cf_embeddings.py) into RAM and yields
conditioning built from samegal physics + neighbor instrument embeddings — never
from the target image itself.

cond_style:
  tokens    -> (1+K, D) sequence [gal(samegal), inst(n1), ..., inst(nK)]; pad rows zeroed
  meanpool  -> concat(gal, masked_mean(inst))  (dim = 2*D = 128)
"""

from __future__ import annotations

import h5py
import torch
from torch.utils.data import Dataset, random_split

_COND_STYLES = ("tokens", "meanpool")


class CFEmbeddingDataset(Dataset):
    def __init__(self, h5_path: str, cond_style: str = "tokens", require_complete: bool = True):
        if cond_style not in _COND_STYLES:
            raise ValueError(f"cond_style must be one of {_COND_STYLES}, got {cond_style!r}")
        self.cond_style = cond_style
        with h5py.File(h5_path, "r") as f:
            complete = bool(f.attrs.get("complete", False))
            if require_complete and not complete:
                raise RuntimeError(
                    f"{h5_path} is missing attrs['complete']=True, so it was not written to "
                    f"completion (a killed precompute leaves pre-allocated ZERO rows that look "
                    f"like valid data: zero targets, zero embeddings, all-False masks). "
                    f"Re-run precompute_cf_embeddings.py, or pass require_complete=False if you "
                    f"really mean to read a truncated file."
                )
            self.emb_galaxy = torch.from_numpy(f["emb_galaxy"][:]).float()
            self.emb_instrument = torch.from_numpy(f["emb_instrument"][:]).float()
            self.masks = torch.from_numpy(f["masks"][:]).bool()
            # Kept at the on-disk dtype (float16 from the ram48-backed store) and cast
            # per item in __getitem__: halves resident RAM, ~1.9 GB instead of ~3.8 GB.
            self.targets = torch.from_numpy(f["targets"][:])
            self.survey = torch.from_numpy(f["survey"][:]).to(torch.int8)
            self.row_idx = torch.from_numpy(f["row_idx"][:]).to(torch.int64)
            self.max_neighbors = int(f.attrs.get("max_neighbors", self.emb_instrument.shape[1]))
            self.emb_dim = int(f.attrs.get("embedding_dim_each", self.emb_galaxy.shape[1]))
            # Anchor count of the source store. Equal to len(self) only for a full
            # corpus, which is the precondition for the seed-42 split to reproduce
            # data_factory's (see is_full_store).
            self.n_anchors_total = int(f.attrs.get("n_anchors_total", 0)) or None
            self.source_loader = str(f.attrs.get("source_loader", "unknown"))
            self.source_data_dir = str(f.attrs.get("source_data_dir", "unknown"))

        assert self.emb_galaxy.shape[0] == self.targets.shape[0]
        self.n_tokens = 1 + self.max_neighbors
        # Architecture dims exposed for the LightningModule constructor.
        if cond_style == "tokens":
            self.cond_dim = self.emb_dim          # per-token dim (= cross_attention_dim)
            self.vector_cond_dim = 2 * self.emb_dim
        else:
            self.cond_dim = 2 * self.emb_dim      # AdaGN vector dim
            self.vector_cond_dim = self.cond_dim

        print(
            f"[CFEmbeddingDataset] {len(self):,} examples | emb_dim={self.emb_dim} "
            f"| K={self.max_neighbors} | cond_style={cond_style} "
            f"| store={self.source_loader} | full={self.is_full_store}"
        )

    @property
    def is_full_store(self) -> bool:
        """True when every anchor of the source store is present, in order.

        Only then does make_train_val_split(0.05, 42) reproduce the split that
        data_factory hands the paper / fm-control models — which is what keeps the
        matched recon-MSE comparison honest.
        """
        if self.n_anchors_total is None:
            return False
        return len(self) == self.n_anchors_total

    def __len__(self) -> int:
        return self.targets.shape[0]

    def _build_cond(self, idx: int):
        g = self.emb_galaxy[idx]                          # (D,)
        I = self.emb_instrument[idx]                      # (K, D)
        m = self.masks[idx]                               # (K,)
        I = I * m.unsqueeze(-1).float()                   # zero pads (already zeroed in H5)

        if self.cond_style == "tokens":
            tokens = torch.cat([g.unsqueeze(0), I], dim=0)  # (1+K, D)
            tok_mask = torch.cat(
                [torch.ones(1, dtype=torch.bool), m], dim=0
            )
            return tokens, tok_mask

        # meanpool
        if m.any():
            inst = (I * m.unsqueeze(-1).float()).sum(dim=0) / m.float().sum().clamp_min(1.0)
        else:
            inst = torch.zeros_like(g)
        cond = torch.cat([g, inst], dim=0)  # (2D,)
        return cond, None

    def __getitem__(self, idx: int):
        cond, tok_mask = self._build_cond(idx)
        survey = "hsc" if int(self.survey[idx]) == 0 else "legacy"
        meta = {
            "anchor_survey": survey,
            "idx": int(self.row_idx[idx]),
            "dataset_index": int(idx),
        }
        target = self.targets[idx].float()
        if tok_mask is not None:
            return cond, tok_mask, target, meta
        return cond, target, meta


def cf_collate(batch):
    """Collate to (cond, [tok_mask|None], targets, metadata).

    tokens mode batch items are (tokens, tok_mask, target, meta).
    meanpool mode items are (cond_vec, target, meta).
    """
    if len(batch[0]) == 4:
        conds = torch.stack([b[0] for b in batch])
        tok_masks = torch.stack([b[1] for b in batch])
        targets = torch.stack([b[2] for b in batch])
        metadata = [b[3] for b in batch]
        return conds, tok_masks, targets, metadata
    conds = torch.stack([b[0] for b in batch])
    targets = torch.stack([b[1] for b in batch])
    metadata = [b[2] for b in batch]
    return conds, None, targets, metadata


def make_train_val_split(dataset: Dataset, val_ratio: float = 0.05, seed: int = 42):
    """Match data_factory.build_neighbors_dataloaders (val_ratio=0.05, seed=42)."""
    n = len(dataset)
    n_val = int(n * val_ratio)
    n_train = n - n_val
    return random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(seed),
    )


def build_catalog_index(dataset: CFEmbeddingDataset) -> dict:
    """Map (row_idx, survey_code) -> h5/dataset index for eval joins.

    Keys must be unique: each catalog row appears exactly once as an anchor and its
    survey is fixed by position parity. A collision means the H5 is not what it
    claims — e.g. unwritten rows all carrying row_idx=0/survey=0 and silently
    aliasing onto the genuine row 0.
    """
    index = {}
    for i in range(len(dataset)):
        key = (int(dataset.row_idx[i]), int(dataset.survey[i]))
        if key in index:
            raise RuntimeError(
                f"duplicate (row_idx, survey) key {key} at dataset rows {index[key]} and {i}; "
                f"the CF H5 is corrupt or truncated — regenerate it."
            )
        index[key] = i
    return index
