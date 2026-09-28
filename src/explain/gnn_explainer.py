"""GNNExplainer for the edge-classification detector.

Given the computation subgraph of one flagged transaction, learn a soft mask over the
neighbouring edges that keeps the seed edge's illicit score while staying sparse. The learned
mask is a per-edge importance: the transactions the model actually relied on to flag this one.
Those importances are what the fidelity metric checks against the true laundering pattern, and
what the demo shows as the "why flagged" subgraph.

This is a self-contained implementation (Ying et al., 2019) rather than a call into
torch_geometric.explain, so it stays pinned to our edge-level head and reverse-MP model and
doesn't break across PyG versions.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from src.explain.masking import EdgeMaskContext
from src.explain.subgraph import EdgeSubgraph

EPS = 1e-6


@dataclass
class ExplanationConfig:
    epochs: int = 200
    lr: float = 0.01
    edge_size_coef: float = 0.008   # sparsity: penalize the SUM of the mask (see note below)
    edge_entropy_coef: float = 0.1  # push masks toward 0/1 (crisp, not grey)


def explain_edge(
    model: torch.nn.Module, sub: EdgeSubgraph, cfg: ExplanationConfig | None = None,
    device: str = "cpu",
) -> torch.Tensor:
    """Return a length-E_sub importance in [0, 1] per subgraph edge (higher = more relied on).

    The seed edge is held at importance 1 (it is the transaction being explained, not evidence
    about itself) so sparsity pressure falls on the *neighbourhood*.
    """
    cfg = cfg or ExplanationConfig()
    sub = sub.to(device)
    model.eval()

    E = sub.edge_index.size(1)
    ctx = EdgeMaskContext(model)
    try:
        with torch.no_grad():
            ctx.set(None)
            target = (_seed_logit(model, sub) > 0).float().detach()

        mask_logits = torch.nn.Parameter(torch.randn(E, device=device) * 0.1)
        opt = torch.optim.Adam([mask_logits], lr=cfg.lr)

        for _ in range(cfg.epochs):
            opt.zero_grad()
            m = torch.sigmoid(mask_logits)
            m = _pin_seed(m, sub.seed_col)
            ctx.set(m)
            logit = _seed_logit(model, sub)
            loss = F.binary_cross_entropy_with_logits(logit.view(1), target.view(1))
            # Sparsity penalizes the SUM (not mean) of the mask, so the per-edge pressure is
            # coef regardless of subgraph size — a fixed coef transfers from small synthetic
            # subgraphs to the larger, hub-heavy ones on real data. The seed-prediction (BCE)
            # term protects the edges the score genuinely needs from being zeroed out.
            loss = loss + cfg.edge_size_coef * m.sum() + cfg.edge_entropy_coef * _entropy(m)
            loss.backward()
            opt.step()

        with torch.no_grad():
            imp = _pin_seed(torch.sigmoid(mask_logits), sub.seed_col).detach().cpu()
    finally:
        ctx.remove()
    return imp


def _seed_logit(model: torch.nn.Module, sub: EdgeSubgraph) -> torch.Tensor:
    return model(sub.x, sub.edge_index, sub.edge_attr)[sub.seed_col]


def _pin_seed(m: torch.Tensor, seed_col: int) -> torch.Tensor:
    pinned = m.clone()
    pinned[seed_col] = 1.0
    return pinned


def _entropy(m: torch.Tensor) -> torch.Tensor:
    ent = -m * torch.log(m + EPS) - (1 - m) * torch.log(1 - m + EPS)
    return ent.mean()
