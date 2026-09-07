"""Mini-batch subgraph sampling for edge classification (enables ego IDs + scaling).

We classify transactions (edges). For a batch of target edges we sample a k-hop
neighbourhood around each with PyG's LinkNeighborLoader (disjoint=True, so every target
gets its OWN subgraph). That per-target subgraph is exactly what ego IDs need: we mark
the two seed endpoints (sender, receiver of the transaction being predicted) with a 1 in
an extra node-feature column, 0 elsewhere.

The seed transaction's own edge features (amount, currency, ports, ...) are looked up from
the full graph via `batch.input_id`, so the readout head sees them regardless of whether
that edge happened to be sampled as a message-passing edge.
"""
from __future__ import annotations

import torch
from torch_geometric.loader import LinkNeighborLoader


def build_link_loader(g, seed_indices: torch.Tensor, num_neighbors, batch_size: int,
                      shuffle: bool):
    """LinkNeighborLoader over the transactions selected by `seed_indices` (into g's edges)."""
    edge_label_index = g.edge_index[:, seed_indices]
    edge_label = g.y[seed_indices]
    return LinkNeighborLoader(
        g,
        num_neighbors=list(num_neighbors),
        edge_label_index=edge_label_index,
        edge_label=edge_label,
        batch_size=batch_size,
        shuffle=shuffle,
        disjoint=True,          # one subgraph per target -> clean ego IDs
        neg_sampling_ratio=0.0,  # we have real labels; do not synthesize negatives
    )


def append_ego_ids(x: torch.Tensor, edge_label_index: torch.Tensor) -> torch.Tensor:
    """Add a [N,1] column that is 1 for the batch's seed endpoints, else 0."""
    ego = torch.zeros((x.size(0), 1), dtype=x.dtype, device=x.device)
    ego[edge_label_index.reshape(-1)] = 1.0
    return torch.cat([x, ego], dim=1)
