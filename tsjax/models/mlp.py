"""MLP model using Flax NNX."""

from __future__ import annotations

import jax
from flax import nnx


class MLP(nnx.Module):
    """Feedforward network for tabular data (normalized-space in, normalized-space out).

    Wrap with :class:`NormalizedModel` or ``nnx.Sequential(Normalize, MLP, Denormalize)``
    to handle raw physical values.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_sizes: list[int] | None = None,
        *,
        rngs: nnx.Rngs,
    ):
        if hidden_sizes is None:
            hidden_sizes = [64, 32]

        # Build hidden layers
        layers = []
        in_feat = input_size
        for h in hidden_sizes:
            layers.append(nnx.Linear(in_features=in_feat, out_features=h, rngs=rngs))
            in_feat = h
        self.hidden_layers = nnx.List(layers)

        # Output projection
        self.linear = nnx.Linear(in_features=in_feat, out_features=output_size, rngs=rngs)

    def __call__(self, u):
        """Forward pass: normalized input -> normalized output.

        u: (batch, input_size)
        returns: (batch, output_size)
        """
        x = u
        for layer in self.hidden_layers:
            x = jax.nn.relu(layer(x))
        return self.linear(x)
