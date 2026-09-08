"""Baseline GIN edge classifier.

Node message passing with GINConv produces account embeddings; a per-edge head reads
out from (src_embedding || dst_embedding || edge_features) to predict whether the
transaction is laundering. This is the Week-1 baseline Person 2 owns; Multi-GIN
(reverse MP + ports + ego ids) subclasses/extends this in Week 2.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv


def _mlp(in_dim: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, out_dim),
        nn.BatchNorm1d(out_dim),
        nn.ReLU(),
        nn.Linear(out_dim, out_dim),
        nn.ReLU(),
    )


class GINEdgeClassifier(nn.Module):
    def __init__(self, node_in: int, edge_in: int, hidden: int = 64,
                 num_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        self.dropout = dropout
        self.input_proj = nn.Linear(node_in, hidden)
        self.convs = nn.ModuleList(
            GINConv(_mlp(hidden, hidden), train_eps=True) for _ in range(num_layers)
        )
        self.edge_head = nn.Sequential(
            nn.Linear(2 * hidden + edge_in, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr=None) -> torch.Tensor:
        # edge_attr accepted for a uniform encode() signature with Multi-GIN (GINConv does
        # not use edge features in message passing); it is ignored here.
        h = self.input_proj(x)
        for conv in self.convs:
            h = conv(h, edge_index)
            h = F.dropout(h, p=self.dropout, training=self.training)
        return h

    def forward(self, x, edge_index, edge_attr) -> torch.Tensor:
        h = self.encode(x, edge_index)
        src, dst = edge_index
        edge_repr = torch.cat([h[src], h[dst], edge_attr], dim=1)
        return self.edge_head(edge_repr).squeeze(-1)  # logits, shape [n_edges]


def build_model(cfg: dict, node_in: int, edge_in: int) -> nn.Module:
    m = cfg["model"]
    if m["arch"] == "gin":
        return GINEdgeClassifier(
            node_in=node_in, edge_in=edge_in,
            hidden=m["hidden_dim"], num_layers=m["num_layers"], dropout=m["dropout"],
        )
    if m["arch"] == "multi_gin":
        from src.models.multi_gin import MultiGINEdgeClassifier

        return MultiGINEdgeClassifier(
            node_in=node_in, edge_in=edge_in,
            hidden=m["hidden_dim"], num_layers=m["num_layers"], dropout=m["dropout"],
            use_reverse_mp=m.get("use_reverse_mp", True),
        )
    raise ValueError(f"Unknown model.arch: {m['arch']}")
