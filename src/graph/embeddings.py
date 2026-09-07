"""Load external node embeddings (e.g. Stage-1 self-supervised account representations)
and align them to this pipeline's node indexing.

This is the Stage-1 -> Stage-2 seam (Person 2 x Person 3): Multi-GIN can be initialized
with pretrained account embeddings instead of only structural degree features. Person 3's
Stage-1 output is consumed here in one of two forms:

  * an array/tensor of shape [n_nodes, d] already in THIS pipeline's node-index order, or
  * a dict {"<bank>_<account>": vector} keyed by account id (we reindex it via
    TransactionData.account_ids, and any account with no embedding gets a zero row).

To stay aligned, Person 3 should build the graph with the same loader (so node indices
match) or emit the account-keyed dict form. `TransactionData.account_ids` is the exact
key list (node index i  <->  account_ids[i]).
"""
from __future__ import annotations

import numpy as np


def load_node_embeddings(path: str, account_ids: np.ndarray) -> np.ndarray:
    n = len(account_ids)
    obj = _load(path)

    if isinstance(obj, dict):
        emb = _reindex_by_account(obj, account_ids)
    else:
        emb = np.asarray(obj, dtype=np.float32)
        if emb.shape[0] != n:
            raise ValueError(
                f"embedding rows ({emb.shape[0]}) != number of nodes ({n}); provide an "
                f"index-aligned [n_nodes, d] array or an account-keyed dict.")
    return emb.astype(np.float32)


def _load(path: str):
    if path.endswith(".npy"):
        return np.load(path, allow_pickle=True).item() if _is_pickled_dict(path) else np.load(path)
    if path.endswith((".pt", ".pth")):
        import torch

        obj = torch.load(path, map_location="cpu", weights_only=False)
        return {k: np.asarray(v) for k, v in obj.items()} if isinstance(obj, dict) \
            else np.asarray(obj)
    raise ValueError(f"unsupported embedding file: {path} (use .npy or .pt)")


def _is_pickled_dict(path: str) -> bool:
    arr = np.load(path, allow_pickle=True)
    return arr.dtype == object and arr.ndim == 0


def _reindex_by_account(mapping: dict, account_ids: np.ndarray) -> np.ndarray:
    keys = list(mapping.keys())
    dim = len(np.asarray(mapping[keys[0]]).ravel())
    out = np.zeros((len(account_ids), dim), dtype=np.float32)
    lookup = {str(k): np.asarray(v, dtype=np.float32).ravel() for k, v in mapping.items()}
    for i, acc in enumerate(account_ids):
        v = lookup.get(str(acc))
        if v is not None:
            out[i] = v
    return out
