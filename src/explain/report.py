"""Turn a model + one flagged transaction into a human-readable explanation record.

This is the payload the briefing's Layer 6 / demo promises: for a flagged transaction, the
subgraph of transactions that drove the alert (GNNExplainer), the features of the transaction
itself that pushed the score up (attribution), and — for evaluation — whether that subgraph
recovered the true laundering pattern. Records serialize to JSON for the demo app and the paper's
qualitative examples.
"""
from __future__ import annotations

import numpy as np
import torch

from src.data.loader import TransactionData
from src.explain.attribution import feature_attribution
from src.explain.gnn_explainer import ExplanationConfig, explain_edge
from src.explain.patterns import PatternSet
from src.explain.subgraph import edge_computation_subgraph


def transaction_view(data: TransactionData, edge_id: int, col: dict) -> dict:
    """The raw transaction fields for one edge (edges are in df row order)."""
    row = data.df.iloc[edge_id]
    return {
        "edge_id": int(edge_id),
        "from_account": str(data.account_ids[data.src[edge_id]]),
        "to_account": str(data.account_ids[data.dst[edge_id]]),
        "timestamp": str(row[col["timestamp"]]),
        "amount_paid": _num(row[col["amount_paid"]]),
        "payment_currency": str(row[col["payment_currency"]]),
        "receiving_currency": str(row[col["receiving_currency"]]),
        "payment_format": str(row[col["payment_format"]]),
        "is_laundering": int(row[col["label"]]),
    }


def explanation_record(
    model: torch.nn.Module, g, data: TransactionData, cfg: dict, feature_names: list[str],
    baseline: torch.Tensor, edge_id: int, patterns: PatternSet | None, prob: float,
    num_hops: int, top_edges: int = 8, top_features: int = 6,
    explain_cfg: ExplanationConfig | None = None, max_edges: int | None = 4000,
    attribution_method: str = "occlusion", device: str = "cpu",
) -> dict:
    col = cfg["data"]["columns"]
    sub = edge_computation_subgraph(g, edge_id, num_hops, max_edges=max_edges, seed=edge_id)
    imp = explain_edge(model, sub, cfg=explain_cfg, device=device).numpy()
    contribs = feature_attribution(model, sub, feature_names, baseline,
                                   method=attribution_method, device=device)

    seed_pid = patterns.edge_pattern_id[edge_id] if patterns is not None else -1
    order = np.argsort(-imp)
    order = [c for c in order if c != sub.seed_col][:top_edges]

    edges_out = []
    for c in order:
        oid = int(sub.orig_edge_ids[c])
        in_pattern = bool(patterns is not None and seed_pid >= 0
                          and patterns.edge_pattern_id[oid] == seed_pid)
        edges_out.append({
            **transaction_view(data, oid, col),
            "importance": round(float(imp[c]), 4),
            "in_true_pattern": in_pattern,
        })

    return {
        "seed": transaction_view(data, edge_id, col),
        "illicit_score": round(float(prob), 4),
        "typology": str(patterns.edge_typology[edge_id]) if patterns is not None else "",
        "top_evidence_transactions": edges_out,
        "top_features": [
            {"feature": c.name, "value": round(c.value, 4), "attribution": round(c.attribution, 4)}
            for c in contribs[:top_features]
        ],
        "subgraph_size": {"nodes": int(sub.nodes.numel()), "edges": int(sub.edge_index.size(1))},
    }


def _num(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")
