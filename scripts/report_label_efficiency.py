"""Label-efficiency curve: from-scratch vs Stage-1 pretrained, side by side.

Groups labeleff_* runs by (arm, label%) and reports mean +/- std across seeds. The arm is
read from the run name ("pretrained" -> Stage-1 arm, else from-scratch), so once Person 3's
embeddings are run through run_label_efficiency.sh with EMBEDDING=..., the two curves and
their gap (delta = how much pretraining buys at each label fraction) appear automatically.

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


def _arm(name: str) -> str:
    return "pretrained" if "pretrained" in name else "scratch"


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "results/experiments.csv"
    try:
        rows = list(csv.DictReader(open(path)))
    except FileNotFoundError:
        print(f"no results at {path} — run scripts/run_label_efficiency.sh first")
        return

    runs = [r for r in rows if r.get("experiment", "").startswith("labeleff")]
    if not runs:
        print("no labeleff* rows yet — run scripts/run_label_efficiency.sh first")
        return

    # data[metric][arm][pct] = [values...]
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in runs:
        pct = _f(r.get("label_pct"))
        if pct is None:
            continue
        arm = _arm(r["experiment"])
        for m in ("minority_f1", "pr_auc"):
            v = _f(r.get(m))
            if v is not None:
                data[m][arm][pct].append(v)

    pcts = sorted({p for m in data.values() for a in m.values() for p in a})
    has_pre = any(data[m]["pretrained"] for m in data)

    print(f"Label-efficiency: from-scratch vs Stage-1 pretrained ({len(runs)} runs)")
    if not has_pre:
        print("(only the from-scratch arm has run so far — pretrained columns fill in once "
              "Person 3's embeddings are run via EMBEDDING=... run_label_efficiency.sh)\n")

    _table("Minority-F1", data["minority_f1"], pcts)
    print()
    _table("PR-AUC", data["pr_auc"], pcts)


def _table(title, arms, pcts):
    print(f"== {title} ==")
    print(f"{'label%':>7}  {'scratch':>16}  {'pretrained':>16}  {'Δ (pre-scratch)':>16}")
    print("-" * 62)
    for pct in pcts:
        s, p = arms["scratch"].get(pct, []), arms["pretrained"].get(pct, [])
        delta = f"{mean(p) - mean(s):+.4f}" if s and p else "-"
        print(f"{pct:>7.0f}  {_agg(s):>16}  {_agg(p):>16}  {delta:>16}")


def _agg(vals):
    if not vals:
        return "-"
    if len(vals) == 1:
        return f"{vals[0]:.4f}"
    return f"{mean(vals):.4f}±{pstdev(vals):.4f}"


if __name__ == "__main__":
    main()
