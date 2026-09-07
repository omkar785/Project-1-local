"""Build a PyG directed-multigraph from a TransactionData bundle.

nodes  = accounts
edges  = transactions (parallel edges between the same pair are kept -> multigraph)
target = per-edge Is Laundering  (edge classification)

Multi-GIN extras: port numbering is applied here (edge-level, so it's a data/graph-side
op); reverse message passing lives in the model (model.use_reverse_mp); ego IDs need
mini-batch subgraph sampling and are deferred to the LinkNeighborLoader step.
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

    if gcfg.get("add_ports"):
        # Port numbering: append repeated-transfer counters so parallel edges between the
        # same accounts stay distinguishable (standardized on train stats like the rest).
        from src.graph.multigraph import port_features

        ports = _standardize(port_features(data.src, data.dst, data.timestamps), fit_mask=is_train)
        edge_feats = np.concatenate([edge_feats, ports], axis=1)

    edge_attr = torch.tensor(edge_feats, dtype=torch.float)

    x = _node_features(data, gcfg)
    x = torch.tensor(_standardize(x.numpy(), fit_mask=None), dtype=torch.float)

    g = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    g.num_nodes = data.n_nodes
    g.timestamps = torch.tensor(data.timestamps, dtype=torch.long)

    # per-edge split masks (edge classification -> masks live on edges)
    split = torch.tensor(data.split, dtype=torch.long)
    g.train_mask = split == 0
    g.val_mask = split == 1
    g.test_mask = split == 2

    if gcfg.get("add_ego_ids"):
        raise NotImplementedError(
            "ego IDs mark the seed nodes of a sampled subgraph, which needs mini-batch "
            "LinkNeighborLoader training — deferred to that step. Keep graph.add_ego_ids=false for now."
        )

    return g


def _standardize(feats: np.ndarray, fit_mask: np.ndarray | None) -> np.ndarray:
    """Z-score columns. Fit mean/std on `fit_mask` rows (train) when given, else all rows."""
    fit = feats if fit_mask is None else feats[fit_mask]
    mean = fit.mean(axis=0, keepdims=True)
    std = fit.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    return (feats - mean) / std


def _degree_features(data: TransactionData) -> np.ndarray:
    n = data.n_nodes
    out_deg = np.bincount(data.src, minlength=n).astype(np.float32)
    in_deg = np.bincount(data.dst, minlength=n).astype(np.float32)
    return np.stack([np.log1p(in_deg), np.log1p(out_deg)], axis=1)


def _node_features(data: TransactionData, gcfg: dict) -> torch.Tensor:
    kind = gcfg["node_features"]
    n = data.n_nodes
    if kind == "ones":
        return torch.ones((n, 1), dtype=torch.float)
    if kind == "degree":
        return torch.tensor(_degree_features(data), dtype=torch.float)
    if kind == "embedding":
        # Stage-1 -> Stage-2 seam: initialize nodes with pretrained account embeddings.
        from src.graph.embeddings import load_node_embeddings

        ecfg = gcfg.get("embedding", {})
        emb = load_node_embeddings(ecfg["path"], data.account_ids)
        if ecfg.get("mode", "replace") == "concat":
            emb = np.concatenate([_degree_features(data), emb], axis=1)
        return torch.tensor(emb, dtype=torch.float)
    raise ValueError(f"Unknown node_features: {kind}")
