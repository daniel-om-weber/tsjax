"""Sequence pooling layers."""

from __future__ import annotations

from flax import nnx


class LastPool(nnx.Module):
    """Select the last time step: ``(*batch, seq_len, features) → (*batch, features)``."""

    def __call__(self, x):
        return x[..., -1, :]
