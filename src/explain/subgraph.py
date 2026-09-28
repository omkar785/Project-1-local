"""Extract the computation subgraph a single transaction (edge) is scored from.

A GNN's prediction for one edge depends only on the L-hop neighbourhood of its two
endpoints (L = number of message-passing layers). Explanation therefore runs on that small
subgraph, not the whole 5M-edge graph. Because the Multi-GIN detector passes messages in
*both* directions (reverse MP), the honest receptive field is the L-hop neighbourhood over
the graph treated as undirected; we then keep every original directed edge whose endpoints
both fall inside that node set.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph, to_undirected


@dataclass
class EdgeSubgraph:
    x: torch.Tensor              # [n_sub, d_node]
    edge_index: torch.Tensor     # [2, E_sub] relabeled to 0..n_sub-1
    edge_attr: torch.Tensor      # [E_sub, d_edge]
    seed_col: int                # column of the explained edge within edge_index
    orig_edge_ids: torch.Tensor  # [E_sub] original edge id of each sub edge (for reporting)
    nodes: torch.Tensor          # [n_sub] original node id of each sub node

    def to(self, device) -> "EdgeSubgraph":
        return EdgeSubgraph(
            self.x.to(device), self.edge_index.to(device), self.edge_attr.to(device),
            self.seed_col, self.orig_edge_ids.to(device), self.nodes.to(device),
        )


def edge_computation_subgraph(
    g: Data, edge_id: int, num_hops: int, max_edges: int | None = 4000,
    seed: int = 0,
) -> EdgeSubgraph:
    """Build the L-hop computation subgraph around edge `edge_id`.

    `max_edges` caps the subgraph so a hub account can't blow up runtime/memory; if the
    honest receptive field is larger it is randomly down-sampled, but the seed edge is always
    kept. Returns an EdgeSubgraph with the seed edge's position recorded.
    """
    ei = g.edge_index
    u = int(ei[0, edge_id]); v = int(ei[1, edge_id])

    # Node receptive field: L hops over the undirected graph (captures both money-in and
    # money-out neighbourhoods that reverse MP mixes).
    undirected = to_undirected(ei, num_nodes=g.num_nodes)
    nodes, _, _, _ = k_hop_subgraph(
        [u, v], num_hops, undirected, relabel_nodes=False, num_nodes=g.num_nodes,
    )

    node_mask = torch.zeros(g.num_nodes, dtype=torch.bool)
    node_mask[nodes] = True
    incident = node_mask[ei[0]] & node_mask[ei[1]]
    orig_edge_ids = incident.nonzero(as_tuple=False).view(-1)

    if max_edges is not None and orig_edge_ids.numel() > max_edges:
        orig_edge_ids = _downsample_keep_seed(orig_edge_ids, edge_id, max_edges, seed)
        # nodes actually used = endpoints of the kept edges (plus the seed's own).
        kept = ei[:, orig_edge_ids]
        used = torch.unique(torch.cat([kept.reshape(-1), torch.tensor([u, v])]))
        nodes = used

    relabel = torch.full((g.num_nodes,), -1, dtype=torch.long)
    relabel[nodes] = torch.arange(nodes.numel())
    sub_edge_index = relabel[ei[:, orig_edge_ids]]

    seed_pos = (orig_edge_ids == edge_id).nonzero(as_tuple=True)[0]
    if seed_pos.numel() == 0:
        raise RuntimeError(f"seed edge {edge_id} dropped from its own subgraph")

    return EdgeSubgraph(
        x=g.x[nodes],
        edge_index=sub_edge_index,
        edge_attr=g.edge_attr[orig_edge_ids],
        seed_col=int(seed_pos[0]),
        orig_edge_ids=orig_edge_ids,
        nodes=nodes,
    )


def _downsample_keep_seed(edge_ids: torch.Tensor, seed_id: int, cap: int, seed: int) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(edge_ids.numel(), generator=gen)[:cap]
    kept = edge_ids[perm]
    if (kept == seed_id).any():
        return kept.sort().values
    # ensure the seed edge survives the cap
    kept = torch.cat([kept[:-1], torch.tensor([seed_id])])
    return kept.sort().values
