"""Pretty-print the ablation table from results/experiments.csv.

    python scripts/report_ablation.py [results/experiments.csv]
"""
from __future__ import annotations

import csv
import sys

COLS = [
    ("experiment", 22), ("arch", 10), ("notes", 16), ("reverse_mp", 10), ("ports", 6),
    ("minority_f1", 11), ("pr_auc", 9), ("precision", 9), ("recall", 8),
    ("roc_auc", 9), ("train_seconds", 8),
]


def fmt(key, val):
    if key in ("minority_f1", "pr_auc", "precision", "recall", "roc_auc"):
        try:
            return f"{float(val):.4f}"
        except (ValueError, TypeError):
            return str(val)
    return str(val)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "results/experiments.csv"
    try:
        with open(path) as f:
            rows = list(csv.DictReader(f))
    except FileNotFoundError:
        print(f"no results yet at {path} — run scripts/run_ablation.sh first")
        return

    header = "  ".join(k.ljust(w) for k, w in COLS)
    print(header)
    print("-" * len(header))
    for r in rows:
        print("  ".join(fmt(k, r.get(k, "")).ljust(w) for k, w in COLS))


if __name__ == "__main__":
    main()
