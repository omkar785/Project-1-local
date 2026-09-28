"""Soft edge masking for GNN explanation.

GNNExplainer works by learning a per-edge mask m in [0, 1] and scaling each edge's
*message* by it: an edge the seed prediction does not need can be pushed toward 0 without
changing the score, an edge it relies on cannot. We apply the mask by hooking every
message-passing layer's `message()` output and multiplying it by m — a mechanism that is
independent of the conv type, so it works for both the plain-GIN baseline (GINConv) and the
Multi-GIN detector (GINEConv), including its reverse-MP convs.

The forward/reverse convs are called on the same subgraph edge_index (reverse just flips the
endpoints, keeping column order), so a single mask vector aligns with every layer's messages.
"""
from __future__ import annotations

import torch
from torch_geometric.nn import MessagePassing


class EdgeMaskContext:
    """Register message hooks on all MessagePassing layers of `model`.

    Set `.mask` (a length-E tensor over the current forward's edges) before a forward to
    scale messages; set it to None to run the model unmodified. Always call `.remove()` when
    done so the hooks don't leak into later forwards.
    """

    def __init__(self, model: torch.nn.Module):
        self.mask: torch.Tensor | None = None
        self._handles = [
            m.register_message_forward_hook(self._hook)
            for m in model.modules()
            if isinstance(m, MessagePassing)
        ]

    def _hook(self, module, inputs, output):
        if self.mask is None:
            return output
        m = self.mask
        if m.size(0) != output.size(0):
            # Guard: a mask must match this layer's edge count. Mismatch means the mask was
            # built for a different edge_index than the one being run — fail loudly rather
            # than silently broadcasting garbage.
            raise ValueError(
                f"edge mask length {m.size(0)} != messages {output.size(0)} "
                "(mask built for a different edge_index?)"
            )
        return output * m.view(-1, *([1] * (output.dim() - 1)))

    def set(self, mask: torch.Tensor | None) -> None:
        self.mask = mask

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []

    def __enter__(self) -> "EdgeMaskContext":
        return self

    def __exit__(self, *exc) -> None:
        self.remove()
