"""Core framework types for tsjax."""

from __future__ import annotations

from flax import nnx


class Buffer(nnx.Variable):
    """Non-trainable variable excluded from gradients."""

    pass
