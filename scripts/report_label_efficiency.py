"""Summarize the label-efficiency curve from results/experiments.csv.

Groups the `labeleff_*` runs by label fraction and (if multiple seeds were run) reports
mean +/- std, so the curve doubles as the Week-4 robustness check.

    python scripts/report_label_efficiency.py [results/experiments.csv]
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from statistics import mean, pstdev


def _f(x):
    try:
        return float(x)
    except (ValueError, TypeError):
        return None


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "results/experiments.csv"
    try:
        rows = list(csv.DictReader(open(path)))
    except FileNotFoundError:
        print(f"no results at {path} — run scripts/run_label_efficiency.sh first")
        return

    runs = [r for r in rows if r.get("experiment", "").startswith("labeleff_")]
    if not runs:
        print("no labeleff_* rows yet — run scripts/run_label_efficiency.sh first")
        return

    by_pct = defaultdict(lambda: defaultdict(list))
    for r in runs:
        pct = _f(r.get("label_pct"))
        if pct is None:
            continue
        for m in ("minority_f1", "pr_auc", "precision", "recall", "roc_auc"):
            v = _f(r.get(m))
            if v is not None:
                by_pct[pct][m].append(v)

    print(f"Label-efficiency curve  ({sum(len(v['minority_f1']) for v in by_pct.values())} runs, "
          f"from {path})")
    print(f"{'label%':>7}  {'seeds':>5}  {'F1':>16}  {'PR-AUC':>16}  {'recall':>16}")
    print("-" * 70)
    for pct in sorted(by_pct):
        d = by_pct[pct]
        print(f"{pct:>7.0f}  {len(d['minority_f1']):>5}  "
              f"{_agg(d['minority_f1']):>16}  {_agg(d['pr_auc']):>16}  {_agg(d['recall']):>16}")


def _agg(vals):
    if not vals:
        return "-"
    if len(vals) == 1:
        return f"{vals[0]:.4f}"
    return f"{mean(vals):.4f}±{pstdev(vals):.4f}"


if __name__ == "__main__":
    main()
