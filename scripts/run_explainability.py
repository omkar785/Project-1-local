"""Layer 6 driver — explain the detector's alerts and score explanation fidelity.

Loads a trained checkpoint, finds the transactions it correctly flagged (true positives),
runs GNNExplainer + feature attribution on each, and measures how often the explanation
recovers the true laundering pattern (the paper's Claim 3 / ">=80% of true positives" target).
Writes an aggregate summary, a per-typology breakdown, and human-readable explanation records
for the demo.

    # local CPU smoke (synthetic, tags built in):
    python scripts/run_explainability.py --checkpoint results/checkpoints/keeper_seed42.pt \
        --num-explain 100 --device cpu

    # server, real data + patterns file:
    python scripts/run_explainability.py --checkpoint results/checkpoints/keeper_seed42.pt \
        --set data.source=csv --set data.csv_path=/path/HI-Small_Trans.csv \
        --set data.patterns_path=/path/HI-Small_Patterns.txt \
        --num-explain 500 --device cuda

The checkpoint stores its own training config, so the graph is rebuilt exactly as trained
(node features, ports); --set only overrides data location / device.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import _apply_override
from src.checkpoint import load_checkpoint
from src.data.loader import load
from src.graph.build import build_graph, edge_feature_names
from src.explain.fidelity import evaluate_fidelity
from src.explain.gnn_explainer import ExplanationConfig
from src.explain.patterns import load_patterns
from src.explain.report import explanation_record
from src.utils.seed import resolve_device, set_seed


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, help="a keeper .pt from train.py (save_checkpoint=true)")
    p.add_argument("--set", dest="overrides", action="append", default=[],
                   help="override the checkpoint's config, e.g. --set data.csv_path=/path.csv")
    p.add_argument("--num-explain", type=int, default=100, help="how many top true positives to explain")
    # These default to None and fall back to the config's `explain:` section (see configs/default.yaml).
    p.add_argument("--num-hops", type=int, default=None, help="receptive field (default: config / model num_layers)")
    p.add_argument("--coverage-threshold", type=float, default=None,
                   help="pattern-recall bar for 'correctly identified'")
    p.add_argument("--max-edges", type=int, default=None, help="cap subgraph size (hub guard)")
    p.add_argument("--num-records", type=int, default=12, help="explanation records to save for the demo")
    p.add_argument("--attribution", default=None, choices=["occlusion", "shap"])
    p.add_argument("--explain-epochs", type=int, default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out-dir", default="results")
    p.add_argument("--tag", default="explain", help="prefix for the output files")
    return p.parse_args()


def main():
    args = parse_args()
    device = resolve_device(args.device)

    model, ckpt = load_checkpoint(args.checkpoint, device=device)
    cfg = ckpt["cfg"]
    for ov in args.overrides:
        _apply_override(cfg, ov)
    set_seed(cfg["experiment"]["seed"])

    # Resolve run knobs: CLI flag wins, else the config's explain: section, else a built-in.
    ex = cfg.get("explain", {})
    gx = ex.get("gnn_explainer", {})
    num_hops = args.num_hops or ex.get("num_hops") or cfg["model"]["num_layers"]
    coverage_threshold = _first(args.coverage_threshold, ex.get("coverage_threshold"), 0.5)
    max_edges = _first(args.max_edges, ex.get("max_subgraph_edges"), 4000)
    attribution = _first(args.attribution, ex.get("attribution"), "occlusion")
    explain_cfg = ExplanationConfig(
        epochs=_first(args.explain_epochs, gx.get("epochs"), 200),
        lr=gx.get("lr", 0.01),
        edge_size_coef=gx.get("edge_size_coef", 0.008),
        edge_entropy_coef=gx.get("edge_entropy_coef", 0.1),
    )

    data = load(cfg)
    g = build_graph(data, cfg)
    _assert_dims(g, ckpt)

    feature_names = edge_feature_names(data, cfg)
    baseline = g.edge_attr[g.train_mask].mean(0)

    tp_ids, tp_probs = _true_positives(model, g, ckpt["threshold"], device)
    if len(tp_ids) == 0:
        print("[warn] no true positives at the checkpoint threshold — nothing to explain.")
        return
    keep = tp_ids[: args.num_explain]
    keep_probs = {int(e): float(p) for e, p in zip(tp_ids, tp_probs)}
    print(f"[setup] {len(tp_ids)} test true positives; explaining top {len(keep)} "
          f"(hops={num_hops}, device={device})")

    patterns = load_patterns(cfg, data)
    if patterns is None:
        print("[warn] no ground-truth patterns (no synthetic tags, no patterns file) — "
              "records will be written but fidelity cannot be scored.")
    else:
        print(f"[patterns] source={patterns.source} labelled-edge coverage={patterns.coverage():.4f}")

    if patterns is not None:
        report = evaluate_fidelity(
            model, g, patterns, [int(e) for e in keep], num_hops,
            coverage_threshold=coverage_threshold, explain_cfg=explain_cfg,
            max_edges=max_edges, device=device, progress=True)
        print("\n" + report.summary())
        _write_summary(args, report)

    records = _build_records(model, g, data, cfg, feature_names, baseline, keep, keep_probs,
                             patterns, num_hops, explain_cfg, max_edges, attribution, args, device)
    _write_records(args, records)


def _first(*vals):
    """First value that isn't None (CLI flag -> config -> built-in default)."""
    for v in vals:
        if v is not None:
            return v
    return None


def _true_positives(model, g, threshold, device):
    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(g.x.to(device), g.edge_index.to(device),
                                    g.edge_attr.to(device))).cpu().numpy()
    test = g.test_mask.numpy()
    y = g.y.numpy()
    tp = np.where(test & (y == 1) & (probs >= threshold))[0]
    order = tp[np.argsort(-probs[tp])]              # most-confident first
    return order, probs[order]


def _build_records(model, g, data, cfg, feature_names, baseline, keep, keep_probs, patterns,
                   num_hops, explain_cfg, max_edges, attribution, args, device):
    records = []
    for e in keep[: args.num_records]:
        e = int(e)
        records.append(explanation_record(
            model, g, data, cfg, feature_names, baseline, e, patterns, keep_probs[e],
            num_hops, explain_cfg=explain_cfg, max_edges=max_edges,
            attribution_method=attribution, device=device))
    return records


def _assert_dims(g, ckpt):
    if g.edge_attr.size(1) != ckpt["edge_in"]:
        raise SystemExit(
            f"edge-feature mismatch: graph has {g.edge_attr.size(1)}, checkpoint expects "
            f"{ckpt['edge_in']}. Rebuild data with the checkpoint's graph settings "
            "(add_ports etc. are read from the checkpoint config).")
    if g.x.size(1) != ckpt["node_in"]:
        raise SystemExit(
            f"node-feature mismatch: graph has {g.x.size(1)}, checkpoint expects {ckpt['node_in']}.")


def _write_summary(args, report):
    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, f"{args.tag}_fidelity.json")
    payload = {
        "coverage_threshold": report.coverage_threshold,
        "n_true_positives": report.n_true_positives,
        "n_scorable": report.n_scorable,
        "fraction_correct": report.fraction_correct,
        "mean_pattern_recall": report.mean_pattern_recall,
        "mean_random_recall": report.mean_random_recall,
        "by_typology": report.by_typology,
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[logged] {path}")


def _write_records(args, records):
    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, f"{args.tag}_records.json")
    with open(path, "w") as f:
        json.dump(records, f, indent=2)
    print(f"[logged] {path}  ({len(records)} explanation records for the demo)")


if __name__ == "__main__":
    main()
