"""Minimal config loader with dotted-key CLI overrides.

Usage:
    cfg = load_config("configs/default.yaml", overrides=["train.epochs=5", "data.source=csv"])
    cfg["train"]["epochs"]  -> 5
"""
from __future__ import annotations

import ast
import copy
from typing import Any, Iterable

import yaml


def load_config(path: str, overrides: Iterable[str] | None = None) -> dict:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    for ov in overrides or []:
        _apply_override(cfg, ov)
    return cfg


def _apply_override(cfg: dict, override: str) -> None:
    if "=" not in override:
        raise ValueError(f"Override must be key=value, got: {override!r}")
    key, raw = override.split("=", 1)
    node: Any = cfg
    parts = key.split(".")
    for p in parts[:-1]:
        if p not in node or not isinstance(node[p], dict):
            node[p] = {}
        node = node[p]
    node[parts[-1]] = _coerce(raw)


def _coerce(raw: str) -> Any:
    """Turn a CLI string into int/float/bool/None/list/str where sensible."""
    low = raw.lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("none", "null"):
        return None
    stripped = raw.strip()
    if stripped.startswith("[") and stripped.endswith("]"):
        try:
            return list(ast.literal_eval(stripped))   # e.g. "[25, 10]" -> [25, 10]
        except (ValueError, SyntaxError):
            pass
    for cast in (int, float):
        try:
            return cast(raw)
        except ValueError:
            pass
    return raw


def deep_copy(cfg: dict) -> dict:
    return copy.deepcopy(cfg)
