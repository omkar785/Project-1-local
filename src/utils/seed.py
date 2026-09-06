"""Reproducibility helpers. Every reported number must be tied to a seed (team rule)."""
from __future__ import annotations

import os
import random

import numpy as np


def set_seed(seed: int, deterministic: bool = True) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def resolve_device(pref: str = "auto") -> str:
    import torch

    if pref == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return pref
