"""Feature attribution on a flagged transaction's own (tabular) features.

GNNExplainer says *which neighbouring transactions* mattered; this says *which features of the
flagged transaction itself* pushed its score up — amount, cross-currency, payment format, the
port/repetition counters. Together they are the "why flagged" the briefing's Layer 6 promises
and the demo surfaces.

Default method is occlusion (replace one feature with its train-set baseline, measure the drop
in the illicit logit) — model-agnostic, dependency-free, and exact for one feature at a time.
An optional KernelSHAP path (method="shap") is used when the `shap` package is installed, for
the Shapley-value attribution the briefing names; it falls back to occlusion otherwise.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from src.explain.subgraph import EdgeSubgraph


@dataclass
class FeatureContribution:
    name: str
    value: float          # standardized feature value on this transaction
    attribution: float    # signed logit change attributed to the feature (+ = toward illicit)


def feature_attribution(
    model: torch.nn.Module, sub: EdgeSubgraph, feature_names: list[str],
    baseline: torch.Tensor, method: str = "occlusion", device: str = "cpu",
    nsamples: int = 200,
) -> list[FeatureContribution]:
    """Attribute the seed edge's logit to its own edge features, sorted by |attribution|.

    `baseline` is the reference feature vector (typically the train-split mean) each feature is
    occluded to. Structure is held fixed — only the seed edge's feature row is perturbed — so
    every forward is a cheap subgraph pass.
    """
    sub = sub.to(device)
    baseline = baseline.to(device)
    model.eval()

    if method == "shap":
        attrs = _shap_attribution(model, sub, baseline, nsamples, device)
        if attrs is not None:
            return _pack(sub, feature_names, attrs)
    attrs = _occlusion_attribution(model, sub, baseline, device)
    return _pack(sub, feature_names, attrs)


def _seed_logit_with_row(model, sub: EdgeSubgraph, row: torch.Tensor) -> float:
    ea = sub.edge_attr.clone()
    ea[sub.seed_col] = row
    with torch.no_grad():
        return float(model(sub.x, sub.edge_index, ea)[sub.seed_col])


def _occlusion_attribution(model, sub, baseline, device) -> np.ndarray:
    seed_row = sub.edge_attr[sub.seed_col]
    base_logit = _seed_logit_with_row(model, sub, seed_row)
    d = seed_row.numel()
    attrs = np.zeros(d, dtype=float)
    for j in range(d):
        occluded = seed_row.clone()
        occluded[j] = baseline[j]
        attrs[j] = base_logit - _seed_logit_with_row(model, sub, occluded)
    return attrs


def _shap_attribution(model, sub, baseline, nsamples, device) -> np.ndarray | None:
    try:
        import shap  # optional dependency
    except Exception:
        return None
    seed_row = sub.edge_attr[sub.seed_col].cpu().numpy()

    def f(rows: np.ndarray) -> np.ndarray:
        out = np.empty(len(rows), dtype=float)
        for i, r in enumerate(rows):
            out[i] = _seed_logit_with_row(model, sub, torch.tensor(r, dtype=sub.edge_attr.dtype, device=device))
        return out

    explainer = shap.KernelExplainer(f, baseline.cpu().numpy().reshape(1, -1))
    vals = explainer.shap_values(seed_row.reshape(1, -1), nsamples=nsamples, silent=True)
    return np.asarray(vals).reshape(-1)


def _pack(sub, feature_names, attrs) -> list[FeatureContribution]:
    seed_row = sub.edge_attr[sub.seed_col].cpu().numpy()
    contribs = [
        FeatureContribution(name=feature_names[j], value=float(seed_row[j]), attribution=float(attrs[j]))
        for j in range(len(attrs))
    ]
    contribs.sort(key=lambda c: abs(c.attribution), reverse=True)
    return contribs
