"""Build a PyG directed-multigraph from a TransactionData bundle.

nodes  = accounts
edges  = transactions (parallel edges between the same pair are kept -> multigraph)
target = per-edge Is Laundering  (edge classification)

Multi-GIN extras (reverse edges, ports, ego ids) are stubbed here and switched on in
Week 2 via the graph.* config flags. For the Week-1 baseline they stay off.
"""
from __future__ import annotations

import numpy as np
import torch
from torch_geometric.data import Data

from src.data.loader import TransactionData


def build_graph(data: TransactionData, cfg: dict) -> Data:
    gcfg = cfg["graph"]

    edge_index = torch.tensor(np.stack([data.src, data.dst]), dtype=torch.long)
    y = torch.tensor(data.labels, dtype=torch.long)

    # Standardize edge features using TRAIN-split statistics only (no leakage from val/test).
    is_train = data.split == 0
    edge_feats = _standardize(data.edge_features, fit_mask=is_train)
    edge_attr = torch.tensor(edge_feats, dtype=torch.float)

    x = _node_features(data, gcfg["node_features"])
    x = torch.tensor(_standardize(x.numpy(), fit_mask=None), dtype=torch.float)

    g = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    g.num_nodes = data.n_nodes
    g.timestamps = torch.tensor(data.timestamps, dtype=torch.long)

    # per-edge split masks (edge classification -> masks live on edges)
    split = torch.tensor(data.split, dtype=torch.long)
    g.train_mask = split == 0
    g.val_mask = split == 1
    g.test_mask = split == 2

    if gcfg.get("add_reverse_edges") or gcfg.get("add_ports") or gcfg.get("add_ego_ids"):
        # Placeholder: Multi-GIN augmentations land in Week 2 (src/graph/multigraph.py).
        raise NotImplementedError(
            "reverse edges / ports / ego ids are a Week-2 Multi-GIN feature; keep these flags false for the baseline."
        )

    return g


def _standardize(feats: np.ndarray, fit_mask: np.ndarray | None) -> np.ndarray:
    """Z-score columns. Fit mean/std on `fit_mask` rows (train) when given, else all rows."""
    fit = feats if fit_mask is None else feats[fit_mask]
    mean = fit.mean(axis=0, keepdims=True)
    std = fit.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    return (feats - mean) / std


def _node_features(data: TransactionData, kind: str) -> torch.Tensor:
    n = data.n_nodes
    if kind == "ones":
        return torch.ones((n, 1), dtype=torch.float)
    if kind == "degree":
        out_deg = np.bincount(data.src, minlength=n).astype(np.float32)
        in_deg = np.bincount(data.dst, minlength=n).astype(np.float32)
        feats = np.stack([np.log1p(in_deg), np.log1p(out_deg)], axis=1)
        return torch.tensor(feats, dtype=torch.float)
    raise ValueError(f"Unknown node_features: {kind}")
