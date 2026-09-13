"""Check a Stage-1 embedding file against this pipeline's accounts BEFORE a full run.

Answers the two things that decide whether the Stage-1 -> Stage-2 handoff works:
  1. key-format match — do the embedding's account keys look like our "<bank>_<account>" ids?
  2. coverage — what fraction of our graph's accounts actually have an embedding
     (the rest zero-fill; low coverage usually means Person 3 trained on a smaller --nrows).

    python scripts/check_embedding_coverage.py --embedding data/stage1_week2_v4.pt \
      --set data.source=csv --set data.csv_path="$DATA_CSV"
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from src.config import load_config
from src.data.loader import load


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--embedding", required=True, help=".pt dict {account_key: vector} or .npy array")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--set", dest="overrides", action="append", default=[])
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config, args.overrides)
    accounts = [str(a) for a in load(cfg).account_ids]
    acc_set = set(accounts)

    if args.embedding.endswith(".npy"):
        obj = np.load(args.embedding, allow_pickle=True)
        if getattr(obj, "dtype", None) == object and obj.ndim == 0:
            obj = obj.item()  # a dict pickled inside a .npy
    else:
        obj = torch.load(args.embedding, map_location="cpu", weights_only=False)
    if not isinstance(obj, dict):
        arr = np.asarray(obj)
        print(f"[array embedding] shape={arr.shape}")
        print(f"graph accounts: {len(accounts)}")
        print("→ index-aligned array: coverage is exact ONLY if it was built with our loader "
              "in our node order. Rows != accounts is a mismatch." )
        print(f"{'MATCH' if arr.shape[0] == len(accounts) else 'MISMATCH'}: "
              f"rows={arr.shape[0]} vs accounts={len(accounts)}")
        return

    keys = [str(k) for k in obj.keys()]
    key_set = set(keys)
    dim = len(np.asarray(next(iter(obj.values()))).ravel())
    matched = acc_set & key_set

    print(f"embedding: {len(keys)} keys, dim={dim}")
    print(f"graph:     {len(accounts)} accounts")
    print(f"MATCHED:   {len(matched)}  ({100*len(matched)/len(accounts):.1f}% of graph accounts "
          f"have an embedding; the rest zero-fill)")
    print(f"emb keys not in graph: {len(key_set - acc_set)}")
    print()
    print("sample graph accounts: ", accounts[:3])
    print("sample embedding keys: ", keys[:3])
    if not matched:
        print("\n⚠ 0% match — key FORMAT differs (not just coverage). Compare the samples above: "
              "our keys are '<bank>_<account>' with bank as a plain int. Fix the producer's keying "
              "or tell me the actual format and I'll adapt src/graph/embeddings.py.")
    elif len(matched) < 0.5 * len(accounts):
        print("\n⚠ low coverage — likely trained on a smaller --nrows subset than our full graph. "
              "Either export Stage-1 on the full dataset, or align both arms with data.nrows.")
    else:
        print("\n✓ good — run:  EMBEDDING=<file> bash scripts/run_label_efficiency.sh")


if __name__ == "__main__":
    main()
