"""Train + evaluate an edge-classification detector on the transaction graph.

Full-graph training (fine for HI-Small); neighbourhood sampling is added later if GPU
memory becomes the bottleneck. Run:

    python -m src.train --config configs/default.yaml
    python -m src.train --config configs/default.yaml --set data.source=csv --set experiment.device=cuda
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch
import torch.nn.functional as F

from src.config import load_config
from src.data.loader import load
from src.graph.build import build_graph
from src.metrics import best_threshold, compute_metrics, format_metrics
from src.models.gin import build_model
from src.utils.logging_utils import log_result
from src.utils.seed import resolve_device, set_seed


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--set", dest="overrides", action="append", default=[],
                   help="Override a config key, e.g. --set train.epochs=5")
    p.add_argument("--label-pct", type=float, default=100.0,
                   help="Percent of TRAIN laundering labels to keep (label-efficiency experiments).")
    return p.parse_args()


def subsample_labels(train_mask: torch.Tensor, y: torch.Tensor, pct: float, seed: int) -> torch.Tensor:
    """Keep `pct`% of the TRAIN labels (drop the rest from supervision).

    Label-efficiency setup: we thin how many transactions are labelled for training but
    keep the full graph structure intact. Applies to all train edges (both classes).
    """
    if pct >= 100.0:
        return train_mask
    rng = np.random.default_rng(seed)
    mask = train_mask.clone()
    train_idx = torch.where(train_mask)[0].numpy()
    keep = rng.random(len(train_idx)) < (pct / 100.0)
    drop = train_idx[~keep]
    mask[drop] = False
    return mask


def class_weight(y: torch.Tensor, mask: torch.Tensor, mode) -> torch.Tensor | None:
    if mode != "auto":
        return None if mode in (None, "none") else torch.tensor(float(mode))
    yl = y[mask]
    pos = float((yl == 1).sum())
    neg = float((yl == 0).sum())
    if pos == 0:
        return None
    return torch.tensor(neg / pos)  # pos_weight for BCEWithLogits


def predict(model, g, mask):
    model.eval()
    with torch.no_grad():
        logits = model(g.x, g.edge_index, g.edge_attr)
        prob = torch.sigmoid(logits[mask]).cpu().numpy()
        y = g.y[mask].cpu().numpy()
    return y, prob


def evaluate(model, g, mask, threshold) -> dict:
    y, prob = predict(model, g, mask)
    return compute_metrics(y, prob, threshold)


def main():
    args = parse_args()
    cfg = load_config(args.config, args.overrides)
    set_seed(cfg["experiment"]["seed"])
    device = resolve_device(cfg["experiment"]["device"])

    data = load(cfg)
    g = build_graph(data, cfg).to(device)

    train_mask = subsample_labels(g.train_mask.cpu(), g.y.cpu(), args.label_pct,
                                  cfg["experiment"]["seed"]).to(device)
    pos_weight = class_weight(g.y, train_mask, cfg["train"]["class_weight"])
    if pos_weight is not None:
        pos_weight = pos_weight.to(device)

    model = build_model(cfg, node_in=g.x.size(1), edge_in=g.edge_attr.size(1)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["train"]["lr"],
                           weight_decay=cfg["train"]["weight_decay"])

    print(f"[setup] device={device} nodes={g.num_nodes} edges={g.edge_index.size(1)} "
          f"edge_feat={g.edge_attr.size(1)} train_pos={int(g.y[train_mask].sum())} "
          f"pos_weight={None if pos_weight is None else round(float(pos_weight),2)}")

    best_monitor, best_state, best_epoch, patience = -1.0, None, 0, 0
    monitor_key = cfg["train"]["monitor"].replace("val_", "")
    t0 = time.time()
    epochs_run = 0

    for epoch in range(1, cfg["train"]["epochs"] + 1):
        model.train()
        opt.zero_grad()
        logits = model(g.x, g.edge_index, g.edge_attr)
        loss = F.binary_cross_entropy_with_logits(
            logits[train_mask], g.y[train_mask].float(), pos_weight=pos_weight)
        loss.backward()
        opt.step()
        epochs_run = epoch

        val = evaluate(model, g, g.val_mask, cfg["eval"]["decision_threshold"])
        cur = val[monitor_key]
        cur = -1.0 if cur != cur else cur  # NaN guard
        improved = cur > best_monitor
        if improved:
            best_monitor, best_epoch, patience = cur, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
        print(f"[epoch {epoch:03d}] loss={loss.item():.4f} val: {format_metrics(val)}"
              + ("  *" if improved else ""))
        if patience >= cfg["train"]["early_stop_patience"]:
            print(f"[early stop] no {monitor_key} gain for {patience} epochs")
            break

    train_seconds = time.time() - t0
    if best_state is not None:
        model.load_state_dict(best_state)

    # Tune the decision threshold on validation (primary metric = minority F1), apply to test.
    val_y, val_prob = predict(model, g, g.val_mask)
    thr = best_threshold(val_y, val_prob)
    test = evaluate(model, g, g.test_mask, thr)
    print(f"\n[best epoch {best_epoch}] threshold={thr:.3f} TEST: {format_metrics(test)}")
    print(f"[time] {train_seconds:.1f}s for {epochs_run} epochs")

    reverse_mp = bool(cfg["model"].get("use_reverse_mp")) if cfg["model"]["arch"] == "multi_gin" else False
    path = log_result(cfg["experiment"]["results_dir"], {
        "experiment": cfg["experiment"]["name"], "arch": cfg["model"]["arch"],
        "reverse_mp": reverse_mp, "ports": bool(cfg["graph"].get("add_ports")),
        "seed": cfg["experiment"]["seed"], "label_pct": args.label_pct, "split": "test",
        "threshold": round(thr, 4),
        "train_seconds": round(train_seconds, 1), "epochs_run": epochs_run, "device": device,
        **{k: test.get(k) for k in ("minority_f1", "pr_auc", "precision", "recall", "roc_auc",
                                     "tp", "fp", "fn", "tn", "n_pos", "n")},
    })
    print(f"[logged] {path}")


if __name__ == "__main__":
    main()
