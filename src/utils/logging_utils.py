"""Append-only experiment result logging.

Team rule: every reported number carries model version, label %, seed, and data split.
Each run appends one row to results/experiments.csv so tables/curves are reproducible.
"""
from __future__ import annotations

import csv
import os
from datetime import datetime, timezone

FIELDS = [
    "timestamp", "experiment", "arch", "seed", "label_pct", "split",
    "minority_f1", "pr_auc", "precision", "recall", "roc_auc",
    "tp", "fp", "fn", "tn", "n_pos", "n",
    "train_seconds", "epochs_run", "device", "notes",
]


def log_result(results_dir: str, row: dict) -> str:
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, "experiments.csv")
    row = {**row, "timestamp": datetime.now(timezone.utc).isoformat()}
    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        if write_header:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in FIELDS})
    return path
