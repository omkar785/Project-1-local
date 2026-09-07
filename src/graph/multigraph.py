"""Directed-multigraph augmentations for Multi-GIN (Egressy et al. reproduction).

Port numbering (implemented here, edge-level):
    The GNN otherwise cannot tell apart *repeated* transactions between the same two
    accounts — a classic laundering signal (structuring, layering). Port numbering gives
    each edge a small set of counters that make parallel/repeated transfers distinguishable:

      port_out  : this edge's rank among the sender's outgoing edges, ordered by time
      port_in   : this edge's rank among the receiver's incoming edges, ordered by time
      parallel  : this edge's rank among edges with the SAME (sender, receiver), by time
      pair_count: how many transactions this exact (sender, receiver) pair has

    All are log1p-scaled here; build_graph then standardizes them on train statistics.

Reverse message passing lives in the model (it's a computation, not a feature).
Ego IDs need mini-batch subgraph sampling and are added at that step.
"""
from __future__ import annotations

import numpy as np


def port_features(src: np.ndarray, dst: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
    n = len(src)
    order = np.lexsort((np.arange(n), timestamps))  # stable, by time then insertion

    port_out = _rank_within(src, order)
    port_in = _rank_within(dst, order)

    pair_key = _pair_ids(src, dst)
    parallel = _rank_within(pair_key, order)
    pair_count = np.bincount(pair_key, minlength=pair_key.max() + 1)[pair_key]

    feats = np.stack([port_out, port_in, parallel, pair_count], axis=1).astype(np.float32)
    return np.log1p(feats)


def _rank_within(group: np.ndarray, order: np.ndarray) -> np.ndarray:
    """For each element, its 0-based occurrence index within its group, walking in
    `order` order. Vectorized (no per-edge Python loop) so it scales to millions of edges."""
    n = len(group)
    g_ordered = group[order]                       # groups in the desired walk order
    sort_idx = np.argsort(g_ordered, kind="stable")  # gather same-group items, order preserved
    sorted_g = g_ordered[sort_idx]
    # within-group 0-based index = global position minus the group's first position
    group_start = np.concatenate([[True], sorted_g[1:] != sorted_g[:-1]])
    first_pos = np.maximum.accumulate(np.where(group_start, np.arange(n), 0))
    idx_within_sorted = np.arange(n) - first_pos
    ranks_ordered = np.empty(n, dtype=np.int64)
    ranks_ordered[sort_idx] = idx_within_sorted
    ranks = np.empty(n, dtype=np.int64)
    ranks[order] = ranks_ordered
    return ranks


def _pair_ids(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Contiguous id per unique (src, dst) ordered pair."""
    keys = src.astype(np.int64) * (dst.max() + 1) + dst.astype(np.int64)
    _, inv = np.unique(keys, return_inverse=True)
    return inv.astype(np.int64)
