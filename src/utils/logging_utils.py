"""Append-only experiment result logging.

Team rule: every reported number carries model version, label %, seed, and data split.
Each run appends one row to results/experiments.csv so tables/curves are reproducible.
"""
from __future__ import annotations

import csv
import os
from datetime import datetime, timezone

FIELDS = [
    "timestamp", "experiment", "arch", "reverse_mp", "ports", "seed", "label_pct", "split",
    "minority_f1", "pr_auc", "precision", "recall", "roc_auc", "threshold",
    "tp", "fp", "fn", "tn", "n_pos", "n",
    "train_seconds", "epochs_run", "device", "notes",
]


def log_result(results_dir: str, row: dict) -> str:
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, "experiments.csv")
    row = {**row, "timestamp": datetime.now(timezone.utc).isoformat()}

    existing = _read_existing(path)
    if existing is None:
        # New file (or one whose header already matches): simple append.
        write_header = not os.path.exists(path)
        with open(path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
            if write_header:
                w.writeheader()
            w.writerow({k: row.get(k, "") for k in FIELDS})
    else:
        # Header changed (columns added) -> rewrite the whole file under the new schema,
        # keeping old rows aligned by column NAME (new columns fill blank).
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
            w.writeheader()
            for r in existing + [row]:
                w.writerow({k: r.get(k, "") for k in FIELDS})
    return path


def _read_existing(path: str):
    """Return prior rows if the file exists but its header differs from FIELDS
    (signals a schema migration is needed); None if absent or already current."""
    if not os.path.exists(path):
        return None
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames == FIELDS:
            return None
        return list(reader)
