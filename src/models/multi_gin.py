"""Multi-GIN edge classifier (directed-multigraph AML detector).

Reproduction of the Egressy-et-al. directed-multigraph adaptations on top of GIN, for
transaction (edge) classification:

  * edge-aware message passing (GINEConv) — transaction features flow through the messages,
    not just the readout head (this is the main lift over the plain-GINConv baseline);
  * reverse message passing (model.use_reverse_mp) — a separate conv over reversed edges so
    each account aggregates both money-in and money-out neighbourhoods;
  * port numbering — supplied as extra edge features by src/graph/multigraph.py
    (graph.add_ports), so repeated transfers stay distinguishable;
  * ego IDs — deferred to the mini-batch (LinkNeighborLoader) step.

Ablation ladder (all logged to the same experiments.csv):
    gin  ->  multi_gin(reverse)  ->  multi_gin(reverse+ports)  ->  full multi_gin(+ego, later)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv


def _mlp(in_dim: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, out_dim),
        nn.BatchNorm1d(out_dim),
        nn.ReLU(),
        nn.Linear(out_dim, out_dim),
        nn.ReLU(),
    )


class MultiGINEdgeClassifier(nn.Module):
    def __init__(self, node_in: int, edge_in: int, hidden: int = 64,
                 num_layers: int = 3, dropout: float = 0.1, use_reverse_mp: bool = True):
        super().__init__()
        self.dropout = dropout
        self.use_reverse_mp = use_reverse_mp
        self.input_proj = nn.Linear(node_in, hidden)

        self.fwd_convs = nn.ModuleList(
            GINEConv(_mlp(hidden, hidden), train_eps=True, edge_dim=edge_in) for _ in range(num_layers)
        )
        if use_reverse_mp:
            self.rev_convs = nn.ModuleList(
                GINEConv(_mlp(hidden, hidden), train_eps=True, edge_dim=edge_in) for _ in range(num_layers)
            )
            # combine forward + reverse node states back to `hidden` each layer
            self.combine = nn.ModuleList(nn.Linear(2 * hidden, hidden) for _ in range(num_layers))

        self.edge_head = nn.Sequential(
            nn.Linear(2 * hidden + edge_in, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def encode(self, x, edge_index, edge_attr) -> torch.Tensor:
        h = self.input_proj(x)
        rev_index = edge_index.flip(0) if self.use_reverse_mp else None
        for i, fwd in enumerate(self.fwd_convs):
            h_f = fwd(h, edge_index, edge_attr)
            if self.use_reverse_mp:
                h_r = self.rev_convs[i](h, rev_index, edge_attr)
                h = F.relu(self.combine[i](torch.cat([h_f, h_r], dim=1)))
            else:
                h = h_f
            h = F.dropout(h, p=self.dropout, training=self.training)
        return h

    def forward(self, x, edge_index, edge_attr) -> torch.Tensor:
        h = self.encode(x, edge_index, edge_attr)
        src, dst = edge_index
        edge_repr = torch.cat([h[src], h[dst], edge_attr], dim=1)
        return self.edge_head(edge_repr).squeeze(-1)
