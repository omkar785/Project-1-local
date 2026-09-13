"""Save / reload the trained best model.

Bundles everything needed to reconstruct the detector without retraining — the state dict,
the config, the input dims, and the tuned decision threshold — so explainability, inference,
or the demo can load a keeper checkpoint directly.
"""
from __future__ import annotations

import os

import torch

from src.models.gin import build_model


def save_checkpoint(path: str, model, cfg: dict, node_in: int, edge_in: int,
                    threshold: float, metrics: dict) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "cfg": cfg,
        "node_in": node_in,
        "edge_in": edge_in,
        "threshold": threshold,
        "metrics": metrics,
    }, path)
    return path


def load_checkpoint(path: str, device: str = "cpu"):
    """Return (model in eval mode on `device`, checkpoint dict)."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = build_model(ckpt["cfg"], node_in=ckpt["node_in"], edge_in=ckpt["edge_in"])
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, ckpt
