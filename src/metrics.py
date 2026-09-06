"""Evaluation metrics for imbalanced AML detection.

Accuracy is deliberately NOT primary (severe class imbalance). Headline metrics are
minority-class F1 and PR-AUC, matching the project's reporting section.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)

    out = {
        "minority_f1": _safe(f1_score, y_true, y_pred, pos_label=1, zero_division=0),
        "precision": _safe(precision_score, y_true, y_pred, pos_label=1, zero_division=0),
        "recall": _safe(recall_score, y_true, y_pred, pos_label=1, zero_division=0),
        "pr_auc": _safe(average_precision_score, y_true, y_prob),
        "roc_auc": _safe(roc_auc_score, y_true, y_prob),
        "n_pos": int(y_true.sum()),
        "n": int(len(y_true)),
    }
    tn, fp, fn, tp = _confusion(y_true, y_pred)
    out.update({"tp": tp, "fp": fp, "fn": fn, "tn": tn})
    return out


def _confusion(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return int(tn), int(fp), int(fn), int(tp)


def _safe(fn, y_true, *args, **kwargs):
    """Return NaN instead of raising when a class is absent (e.g. tiny val split)."""
    if len(np.unique(y_true)) < 2 and fn in (average_precision_score, roc_auc_score):
        return float("nan")
    try:
        return float(fn(y_true, *args, **kwargs))
    except ValueError:
        return float("nan")


def best_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Threshold that maximizes minority-class F1 (tuned on validation, applied to test).

    Under heavy class weighting a fixed 0.5 cut is meaningless, so we pick the operating
    point on val. Ties broken toward the higher threshold (fewer false positives).
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    if len(np.unique(y_true)) < 2:
        return 0.5
    cands = np.unique(np.quantile(y_prob, np.linspace(0.5, 0.999, 50)))
    best_t, best_f1 = 0.5, -1.0
    for t in cands:
        f1 = f1_score(y_true, (y_prob >= t).astype(int), pos_label=1, zero_division=0)
        if f1 >= best_f1:
            best_f1, best_t = f1, float(t)
    return best_t


def format_metrics(m: dict) -> str:
    return (
        f"F1={m['minority_f1']:.4f} PR-AUC={m['pr_auc']:.4f} "
        f"P={m['precision']:.4f} R={m['recall']:.4f} ROC-AUC={m['roc_auc']:.4f} "
        f"[tp={m['tp']} fp={m['fp']} fn={m['fn']} tn={m['tn']}] pos={m['n_pos']}/{m['n']}"
    )
