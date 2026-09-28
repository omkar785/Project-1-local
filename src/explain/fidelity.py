"""Explanation-fidelity metric — the paper's Claim 3.

For each correctly flagged transaction (a true positive), GNNExplainer produces an importance
over the transactions in its receptive field. Fidelity asks: did the explanation actually
surface the *rest of the laundering pattern*? Concretely, of the pattern's other edges that lie
within reach, how many are among the explanation's top-scored edges (a budget equal to the
number of pattern edges present).

  pattern_recall = |top-k explanation edges  ∩  pattern edges in subgraph| / |pattern edges in subgraph|

A transaction "correctly identifies its pattern" when pattern_recall >= coverage_threshold. The
headline number is the fraction of scorable true positives that clear that bar — the ">=80% of
true positives" target. Random-baseline recall (budget / candidate edges) is reported alongside,
so the lift over chance is visible and the metric can't be gamed by a dense subgraph.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from src.explain.gnn_explainer import ExplanationConfig, explain_edge
from src.explain.patterns import PatternSet
from src.explain.subgraph import edge_computation_subgraph


@dataclass
class EdgeFidelity:
    edge_id: int
    typology: str
    n_pattern_in_subgraph: int   # pattern edges reachable (excludes the seed itself)
    n_recovered: int             # of those, how many the explanation's top-k caught
    pattern_recall: float
    random_recall: float         # chance level for the same budget
    scorable: bool               # False when no pattern edge is in reach (can't be scored)


@dataclass
class FidelityReport:
    coverage_threshold: float
    n_true_positives: int
    n_scorable: int
    fraction_correct: float          # THE headline: scorable TPs clearing the threshold
    mean_pattern_recall: float
    mean_random_recall: float
    by_typology: dict = field(default_factory=dict)
    per_edge: list = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"explanation fidelity @ recall>={self.coverage_threshold:.2f}: "
            f"{self.fraction_correct:.3f} of {self.n_scorable} scorable TPs "
            f"(mean recall {self.mean_pattern_recall:.3f} vs chance {self.mean_random_recall:.3f})"
        )


def evaluate_fidelity(
    model: torch.nn.Module, g, patterns: PatternSet, tp_edge_ids: list[int], num_hops: int,
    coverage_threshold: float = 0.5, explain_cfg: ExplanationConfig | None = None,
    max_edges: int | None = 4000, device: str = "cpu", progress: bool = False,
) -> FidelityReport:
    per_edge: list[EdgeFidelity] = []
    for i, e in enumerate(tp_edge_ids):
        ef = _fidelity_for_edge(
            model, g, patterns, int(e), num_hops, coverage_threshold, explain_cfg, max_edges, device)
        per_edge.append(ef)
        if progress and (i + 1) % 25 == 0:
            print(f"  explained {i + 1}/{len(tp_edge_ids)} true positives")
    return _aggregate(per_edge, coverage_threshold)


def _fidelity_for_edge(
    model, g, patterns, edge_id, num_hops, coverage_threshold, explain_cfg, max_edges, device,
) -> EdgeFidelity:
    typ = str(patterns.edge_typology[edge_id])
    sub = edge_computation_subgraph(g, edge_id, num_hops, max_edges=max_edges, seed=edge_id)

    # Pattern edges that actually lie within this seed's receptive field (exclude the seed).
    orig = sub.orig_edge_ids.numpy()
    sub_pid = patterns.edge_pattern_id[orig]
    seed_pid = patterns.edge_pattern_id[edge_id]
    is_pattern = (sub_pid == seed_pid) & (orig != edge_id) & (seed_pid >= 0)
    n_pattern = int(is_pattern.sum())

    if n_pattern == 0:
        # The rest of the pattern is out of reach (or the seed is a singleton) — not scorable.
        return EdgeFidelity(edge_id, typ, 0, 0, float("nan"), float("nan"), scorable=False)

    imp = explain_edge(model, sub, cfg=explain_cfg, device=device).numpy()
    # Rank neighbourhood edges by importance, excluding the seed (pinned to 1 by construction).
    order = np.argsort(-imp)
    order = order[order != sub.seed_col]
    topk = set(order[:n_pattern].tolist())          # budget = number of pattern edges present
    pattern_cols = set(np.nonzero(is_pattern)[0].tolist())

    recovered = len(topk & pattern_cols)
    recall = recovered / n_pattern
    n_candidates = len(imp) - 1                      # everything except the seed
    random_recall = n_pattern / n_candidates if n_candidates > 0 else 0.0

    return EdgeFidelity(
        edge_id=edge_id, typology=typ, n_pattern_in_subgraph=n_pattern, n_recovered=recovered,
        pattern_recall=recall, random_recall=float(random_recall), scorable=True,
    )


def _aggregate(per_edge: list[EdgeFidelity], threshold: float) -> FidelityReport:
    scorable = [e for e in per_edge if e.scorable]
    n_scor = len(scorable)
    frac = float(np.mean([e.pattern_recall >= threshold for e in scorable])) if n_scor else 0.0
    mean_recall = float(np.mean([e.pattern_recall for e in scorable])) if n_scor else 0.0
    mean_rand = float(np.mean([e.random_recall for e in scorable])) if n_scor else 0.0

    by_typ: dict[str, dict] = {}
    for e in scorable:
        b = by_typ.setdefault(e.typology, {"n": 0, "correct": 0, "recall_sum": 0.0})
        b["n"] += 1
        b["correct"] += int(e.pattern_recall >= threshold)
        b["recall_sum"] += e.pattern_recall
    for b in by_typ.values():
        b["fraction_correct"] = b["correct"] / b["n"]
        b["mean_recall"] = b["recall_sum"] / b["n"]
        del b["recall_sum"]

    return FidelityReport(
        coverage_threshold=threshold, n_true_positives=len(per_edge), n_scorable=n_scor,
        fraction_correct=frac, mean_pattern_recall=mean_recall, mean_random_recall=mean_rand,
        by_typology=by_typ, per_edge=per_edge,
    )
