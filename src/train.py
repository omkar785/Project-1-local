"""Train + evaluate an edge-classification detector on the transaction graph.

Two training modes (train.mode):
  * full_graph  — one gradient step over the whole graph per epoch. Fast on the H100 for
                  HI-Small; used for the GIN baseline and the reverse-MP / ports ablations.
  * minibatch   — LinkNeighborLoader subgraph sampling. Required for ego IDs (full
                  Multi-GIN) and the scaling / label-efficiency work.

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
                   help="Percent of TRAIN labels to keep (label-efficiency experiments).")
    return p.parse_args()


def subsample_labels(train_mask, y, pct, seed):
    """Keep `pct`% of the TRAIN labels; graph structure stays intact."""
    if pct >= 100.0:
        return train_mask
    rng = np.random.default_rng(seed)
    mask = train_mask.clone()
    train_idx = torch.where(train_mask)[0].numpy()
    drop = train_idx[rng.random(len(train_idx)) >= (pct / 100.0)]
    mask[drop] = False
    return mask


def class_weight(y, mask, mode):
    if mode != "auto":
        return None if mode in (None, "none") else torch.tensor(float(mode))
    yl = y[mask]
    pos, neg = float((yl == 1).sum()), float((yl == 0).sum())
    return None if pos == 0 else torch.tensor(neg / pos)  # pos_weight for BCEWithLogits


# --------------------------------------------------------------------------- full graph

def predict_full(model, g, mask):
    model.eval()
    with torch.no_grad():
        logits = model(g.x, g.edge_index, g.edge_attr)
        return g.y[mask].cpu().numpy(), torch.sigmoid(logits[mask]).cpu().numpy()


def train_full_graph(cfg, g, model, opt, train_mask, pos_weight, monitor_key):
    best_monitor, best_state, best_epoch, patience = -1.0, None, 0, 0
    t0, epochs_run = time.time(), 0
    for epoch in range(1, cfg["train"]["epochs"] + 1):
        model.train()
        opt.zero_grad()
        logits = model(g.x, g.edge_index, g.edge_attr)
        loss = F.binary_cross_entropy_with_logits(
            logits[train_mask], g.y[train_mask].float(), pos_weight=pos_weight)
        loss.backward()
        opt.step()
        epochs_run = epoch

        val = compute_metrics(*predict_full(model, g, g.val_mask), cfg["eval"]["decision_threshold"])
        best_monitor, best_state, best_epoch, patience, stop = _track(
            val, monitor_key, best_monitor, best_state, best_epoch, patience, model, epoch, cfg)
        print(f"[epoch {epoch:03d}] loss={loss.item():.4f} val: {format_metrics(val)}"
              + ("  *" if patience == 0 else ""))
        if stop:
            print(f"[early stop] no {monitor_key} gain for {patience} epochs")
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return best_epoch, epochs_run, time.time() - t0


# --------------------------------------------------------------------------- minibatch

def _run_batches(model, g, loader, seed_index, edge_attr_dev, use_ego, device, opt=None,
                 pos_weight=None):
    """One pass over `loader`. Train step if `opt` given, else collect (y, prob)."""
    from src.sampling import append_ego_ids
    train = opt is not None
    model.train() if train else model.eval()
    ys, ps = [], []
    for batch in loader:
        batch = batch.to(device)
        x = append_ego_ids(batch.x, batch.edge_label_index) if use_ego else batch.x
        # seed_index lives on CPU (it indexes the CPU graph in the loader); map the batch's
        # input_id back through it, then move the original edge ids to device for the lookup.
        seed_ids = seed_index[batch.input_id.cpu()].to(device)
        seed_feats = edge_attr_dev.index_select(0, seed_ids)
        with torch.set_grad_enabled(train):
            h = model.encode(x, batch.edge_index, batch.edge_attr)
            s, d = batch.edge_label_index
            logits = model.edge_head(torch.cat([h[s], h[d], seed_feats], dim=1)).squeeze(-1)
            if train:
                loss = F.binary_cross_entropy_with_logits(
                    logits, batch.edge_label.float(), pos_weight=pos_weight)
                opt.zero_grad(); loss.backward(); opt.step()
            else:
                ys.append(batch.edge_label.cpu().numpy())
                ps.append(torch.sigmoid(logits).cpu().numpy())
    if not train:
        return np.concatenate(ys), np.concatenate(ps)
    return None


def train_minibatch(cfg, g, model, opt, seed_index, edge_attr_dev, pos_weight, monitor_key,
                    use_ego, device):
    from src.sampling import build_link_loader
    tc = cfg["train"]
    train_loader = build_link_loader(g, seed_index, tc["num_neighbors"], tc["batch_size"], shuffle=True)
    val_seed = torch.where(g.val_mask)[0]
    val_loader = build_link_loader(g, val_seed, tc["num_neighbors"], tc["eval_batch_size"], shuffle=False)

    best_monitor, best_state, best_epoch, patience = -1.0, None, 0, 0
    t0, epochs_run = time.time(), 0
    for epoch in range(1, tc["epochs"] + 1):
        _run_batches(model, g, train_loader, seed_index, edge_attr_dev, use_ego, device,
                     opt=opt, pos_weight=pos_weight)
        epochs_run = epoch
        y, p = _run_batches(model, g, val_loader, val_seed, edge_attr_dev, use_ego, device)
        val = compute_metrics(y, p, cfg["eval"]["decision_threshold"])
        best_monitor, best_state, best_epoch, patience, stop = _track(
            val, monitor_key, best_monitor, best_state, best_epoch, patience, model, epoch, cfg)
        print(f"[epoch {epoch:03d}] val: {format_metrics(val)}" + ("  *" if patience == 0 else ""))
        if stop:
            print(f"[early stop] no {monitor_key} gain for {patience} epochs")
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return best_epoch, epochs_run, time.time() - t0


def predict_minibatch(cfg, g, model, split_mask, edge_attr_dev, use_ego, device):
    from src.sampling import build_link_loader
    seed = torch.where(split_mask)[0]
    loader = build_link_loader(g, seed, cfg["train"]["num_neighbors"],
                               cfg["train"]["eval_batch_size"], shuffle=False)
    return _run_batches(model, g, loader, seed, edge_attr_dev, use_ego, device)


# --------------------------------------------------------------------------- shared

def _track(val, monitor_key, best_monitor, best_state, best_epoch, patience, model, epoch, cfg):
    cur = val[monitor_key]
    cur = -1.0 if cur != cur else cur  # NaN guard
    if cur > best_monitor:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        return cur, best_state, epoch, 0, False
    patience += 1
    return best_monitor, best_state, best_epoch, patience, patience >= cfg["train"]["early_stop_patience"]


def main():
    args = parse_args()
    cfg = load_config(args.config, args.overrides)
    set_seed(cfg["experiment"]["seed"])
    device = resolve_device(cfg["experiment"]["device"])
    mode = cfg["train"].get("mode", "full_graph")
    use_ego = bool(cfg["model"].get("use_ego_ids"))
    monitor_key = cfg["train"]["monitor"].replace("val_", "")

    data = load(cfg)
    g = build_graph(data, cfg)  # kept on CPU; moved per-mode below

    train_mask = subsample_labels(g.train_mask, g.y, args.label_pct, cfg["experiment"]["seed"])
    pos_weight = class_weight(g.y, train_mask, cfg["train"]["class_weight"])
    if pos_weight is not None:
        pos_weight = pos_weight.to(device)

    node_in = g.x.size(1) + (1 if (mode == "minibatch" and use_ego) else 0)
    model = build_model(cfg, node_in=node_in, edge_in=g.edge_attr.size(1)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["train"]["lr"],
                           weight_decay=cfg["train"]["weight_decay"])

    print(f"[setup] mode={mode} device={device} nodes={g.num_nodes} edges={g.edge_index.size(1)} "
          f"node_feat={node_in} edge_feat={g.edge_attr.size(1)} train_pos={int(g.y[train_mask].sum())} "
          f"pos_weight={None if pos_weight is None else round(float(pos_weight),2)}"
          + (f" batch={cfg['train']['batch_size']} nbrs={cfg['train']['num_neighbors']}"
             if mode == "minibatch" else ""))

    if mode == "full_graph":
        if use_ego:
            raise ValueError("ego IDs require train.mode=minibatch (set model.use_ego_ids=false for full_graph).")
        g = g.to(device)
        train_mask = train_mask.to(device)
        best_epoch, epochs_run, secs = train_full_graph(cfg, g, model, opt, train_mask, pos_weight, monitor_key)
        get_probs = lambda mask: predict_full(model, g, mask)
        val_probs = get_probs(g.val_mask)
        test_probs = get_probs(g.test_mask)
    else:
        if cfg["model"]["arch"] != "multi_gin":
            raise ValueError("minibatch mode is implemented for model.arch=multi_gin.")
        edge_attr_dev = g.edge_attr.to(device)
        seed_index = torch.where(train_mask)[0]
        best_epoch, epochs_run, secs = train_minibatch(
            cfg, g, model, opt, seed_index, edge_attr_dev, pos_weight, monitor_key, use_ego, device)
        val_probs = predict_minibatch(cfg, g, model, g.val_mask, edge_attr_dev, use_ego, device)
        test_probs = predict_minibatch(cfg, g, model, g.test_mask, edge_attr_dev, use_ego, device)

    thr = best_threshold(*val_probs)
    test = compute_metrics(*test_probs, thr)
    print(f"\n[best epoch {best_epoch}] threshold={thr:.3f} TEST: {format_metrics(test)}")
    print(f"[time] {secs:.1f}s for {epochs_run} epochs")

    reverse_mp = bool(cfg["model"].get("use_reverse_mp")) if cfg["model"]["arch"] == "multi_gin" else False
    notes = ";".join(filter(None, [mode, "ego" if use_ego else ""]))
    path = log_result(cfg["experiment"]["results_dir"], {
        "experiment": cfg["experiment"]["name"], "arch": cfg["model"]["arch"],
        "reverse_mp": reverse_mp, "ports": bool(cfg["graph"].get("add_ports")),
        "seed": cfg["experiment"]["seed"], "label_pct": args.label_pct, "split": "test",
        "threshold": round(thr, 4), "notes": notes,
        "train_seconds": round(secs, 1), "epochs_run": epochs_run, "device": device,
        **{k: test.get(k) for k in ("minority_f1", "pr_auc", "precision", "recall", "roc_auc",
                                     "tp", "fp", "fn", "tn", "n_pos", "n")},
    })
    print(f"[logged] {path}")


if __name__ == "__main__":
    main()
