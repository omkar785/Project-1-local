"""Write placeholder node embeddings to exercise the Stage-1 -> Stage-2 seam before
Person 3's real Stage-1 model exists. Also documents the two accepted output formats.

    # index-aligned array [n_nodes, d]  (Person 3 must use the same loader for alignment)
    python scripts/make_fake_embeddings.py --format npy --out data/fake_emb.npy --dim 32

    # account-keyed dict {"<bank>_<account>": vector}  (alignment-agnostic)
    python scripts/make_fake_embeddings.py --format pt  --out data/fake_emb.pt  --dim 32

Then:
    python -m src.train --config configs/default.yaml \
      --set model.arch=multi_gin --set model.use_reverse_mp=true \
      --set graph.node_features=embedding --set graph.embedding.path=data/fake_emb.npy
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.config import load_config
from src.data.loader import load


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--format", choices=["npy", "pt"], default="npy")
    p.add_argument("--out", default="data/fake_emb.npy")
    p.add_argument("--dim", type=int, default=32)
    p.add_argument("--set", dest="overrides", action="append", default=[])
    args = p.parse_args()

    cfg = load_config(args.config, args.overrides)
    data = load(cfg)
    rng = np.random.default_rng(cfg["experiment"]["seed"])
    emb = rng.standard_normal((data.n_nodes, args.dim)).astype(np.float32)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    if args.format == "npy":
        np.save(args.out, emb)
    else:
        import torch

        mapping = {str(acc): emb[i] for i, acc in enumerate(data.account_ids)}
        torch.save(mapping, args.out)
    print(f"wrote {args.format} embeddings [{data.n_nodes}, {args.dim}] -> {args.out}")


if __name__ == "__main__":
    main()
