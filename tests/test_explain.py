"""End-to-end guard for the Layer-6 explainability stack (CPU, synthetic, ~30s).

Runs without pytest:  .venv/bin/python tests/test_explain.py

Checks the invariants that make the fidelity number trustworthy:
  1. the edge mask is a true no-op at all-ones, and removing every edge collapses the seed
     score (the message hook works and the model actually uses structure);
  2. GNNExplainer beats random chance at recovering the ground-truth pattern edges;
  3. feature attribution returns contributions ranked by magnitude;
  4. the AMLworld patterns parser round-trips a written patterns block back to graph edges.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config
from src.data.loader import load
from src.graph.build import build_graph, edge_feature_names
from src.explain.attribution import feature_attribution
from src.explain.gnn_explainer import ExplanationConfig, explain_edge
from src.explain.masking import EdgeMaskContext
from src.explain.patterns import from_amlworld_file, load_patterns
from src.explain.subgraph import edge_computation_subgraph
from src.models.gin import build_model
from src.utils.seed import set_seed

HOPS = 2


def _tiny_setup():
    """A small, sparse synthetic graph + a briefly-trained Multi-GIN detector."""
    set_seed(0)
    cfg = load_config("configs/default.yaml", [
        "data.source=synthetic",
        "data.synthetic.n_accounts=1500", "data.synthetic.n_transactions=5000",
        "data.synthetic.illicit_ratio=0.06",
        "model.arch=multi_gin", "model.use_reverse_mp=true", "model.num_layers=2",
        "graph.add_ports=true",
    ])
    data = load(cfg)
    g = build_graph(data, cfg)
    model = build_model(cfg, node_in=g.x.size(1), edge_in=g.edge_attr.size(1))
    opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    pos = float((g.y[g.train_mask] == 1).sum()); neg = float((g.y[g.train_mask] == 0).sum())
    pos_weight = torch.tensor(neg / max(pos, 1.0))
    for _ in range(60):
        model.train(); opt.zero_grad()
        logits = model(g.x, g.edge_index, g.edge_attr)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits[g.train_mask], g.y[g.train_mask].float(), pos_weight=pos_weight)
        loss.backward(); opt.step()
    model.eval()
    return cfg, data, g, model


def _true_positives(model, g):
    with torch.no_grad():
        probs = torch.sigmoid(model(g.x, g.edge_index, g.edge_attr)).numpy()
    tp = np.where(g.test_mask.numpy() & (g.y.numpy() == 1) & (probs >= 0.5))[0]
    return tp[np.argsort(-probs[tp])]


def test_mask_is_noop_and_structure_matters(cfg, data, g, model):
    tp = _true_positives(model, g)
    assert len(tp) > 0, "model learned nothing on synthetic data — check training"
    sub = edge_computation_subgraph(g, int(tp[0]), HOPS)
    ctx = EdgeMaskContext(model)
    try:
        with torch.no_grad():
            ctx.set(None)
            base = float(model(sub.x, sub.edge_index, sub.edge_attr)[sub.seed_col])
            ones = torch.ones(sub.edge_index.size(1))
            ctx.set(ones)
            on = float(model(sub.x, sub.edge_index, sub.edge_attr)[sub.seed_col])
            off = ones.clone(); off[:] = 0.0; off[sub.seed_col] = 1.0
            ctx.set(off)
            none = float(model(sub.x, sub.edge_index, sub.edge_attr)[sub.seed_col])
    finally:
        ctx.remove()
    assert abs(base - on) < 1e-4, f"all-ones mask must be a no-op ({base} vs {on})"
    assert abs(base - none) > 0.5, f"removing all edges should move the seed score ({base} vs {none})"
    print(f"  [ok] mask no-op at 1.0; masking all edges moves logit {base:.2f} -> {none:.2f}")


def test_explainer_beats_chance(cfg, data, g, model):
    patterns = load_patterns(cfg, data)
    assert patterns is not None and patterns.source == "synthetic"
    tp = _true_positives(model, g)[:12]
    recalls, chances = [], []
    for e in tp:
        e = int(e)
        sub = edge_computation_subgraph(g, e, HOPS)
        orig = sub.orig_edge_ids.numpy(); pid = patterns.edge_pattern_id[e]
        pat_cols = np.where((patterns.edge_pattern_id[orig] == pid) & (orig != e))[0]
        if len(pat_cols) == 0:
            continue
        imp = explain_edge(model, sub, ExplanationConfig(epochs=150)).numpy()
        order = np.argsort(-imp); order = order[order != sub.seed_col]
        topk = set(order[:len(pat_cols)].tolist())
        recalls.append(len(topk & set(pat_cols.tolist())) / len(pat_cols))
        chances.append(len(pat_cols) / (len(imp) - 1))
    assert recalls, "no scorable true positives — pattern edges never reachable"
    mean_recall, mean_chance = float(np.mean(recalls)), float(np.mean(chances))
    assert mean_recall > 3 * mean_chance, f"explainer no better than chance ({mean_recall:.3f} vs {mean_chance:.3f})"
    print(f"  [ok] explainer recall {mean_recall:.3f} >> chance {mean_chance:.3f} over {len(recalls)} TPs")


def test_feature_attribution_ranked(cfg, data, g, model):
    names = edge_feature_names(data, cfg)
    baseline = g.edge_attr[g.train_mask].mean(0)
    e = int(_true_positives(model, g)[0])
    sub = edge_computation_subgraph(g, e, HOPS)
    contribs = feature_attribution(model, sub, names, baseline, method="occlusion")
    assert len(contribs) == len(names)
    mags = [abs(c.attribution) for c in contribs]
    assert mags == sorted(mags, reverse=True), "attributions must be sorted by |value|"
    print(f"  [ok] top feature: {contribs[0].name} (attr {contribs[0].attribution:+.3f})")


def test_amlworld_parser_roundtrip(cfg, data, g, model):
    """Write two illicit transactions into a patterns block and parse them back to edges."""
    col = cfg["data"]["columns"]
    illicit = np.where(data.labels == 1)[0][:2]
    lines = ["BEGIN LAUNDERING ATTEMPT - FAN-OUT"]
    for e in illicit:
        row = data.df.iloc[int(e)]
        lines.append(",".join(str(row[col[k]]) for k in [
            "timestamp", "from_bank", "from_account", "to_bank", "to_account",
            "amount_received", "receiving_currency", "amount_paid", "payment_currency",
            "payment_format", "label"]))
    lines.append("END LAUNDERING ATTEMPT - FAN-OUT")
    path = os.path.join(os.path.dirname(__file__), "_tmp_patterns.txt")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    try:
        ps = from_amlworld_file(path, data, col)
        tagged = int((ps.edge_pattern_id >= 0).sum())
        assert tagged >= len(illicit), f"parser matched {tagged} edges, expected >= {len(illicit)}"
        assert set(ps.edge_typology[ps.edge_pattern_id >= 0]) == {"FAN-OUT"}
    finally:
        os.remove(path)
    print(f"  [ok] AMLworld parser matched {tagged} edges back to graph")


def main():
    cfg, data, g, model = _tiny_setup()
    tests = [
        test_mask_is_noop_and_structure_matters,
        test_explainer_beats_chance,
        test_feature_attribution_ranked,
        test_amlworld_parser_roundtrip,
    ]
    for t in tests:
        print(f"- {t.__name__}")
        t(cfg, data, g, model)
    print("\nALL EXPLAINABILITY TESTS PASSED")


if __name__ == "__main__":
    main()
