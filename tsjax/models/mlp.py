"""MLP model with internal normalization using Flax NNX."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx

from tsjax._core import Buffer


class MLP(nnx.Module):
    """Feedforward network for tabular data with internal normalization.

    Raw physical values in, raw physical values out.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_sizes: list[int] | None = None,
        u_mean: jnp.ndarray | None = None,
        u_std: jnp.ndarray | None = None,
        y_mean: jnp.ndarray | None = None,
        y_std: jnp.ndarray | None = None,
        *,
        rngs: nnx.Rngs,
    ):
        if hidden_sizes is None:
            hidden_sizes = [64, 32]

        # Store norm stats as Buffer (excluded from gradients)
        self.u_mean = Buffer(jnp.zeros(input_size) if u_mean is None else jnp.asarray(u_mean))
        self.u_std = Buffer(jnp.ones(input_size) if u_std is None else jnp.asarray(u_std))
        self.y_mean = Buffer(
            jnp.zeros(output_size) if y_mean is None else jnp.asarray(y_mean)
        )
        self.y_std = Buffer(jnp.ones(output_size) if y_std is None else jnp.asarray(y_std))

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
        """Forward pass: raw input -> raw output.

        u: (batch, input_size) — raw physical values
        returns: (batch, output_size) — raw physical values
        """
        # Normalize input
        x = (u - self.u_mean[...]) / self.u_std[...]

        # Hidden layers with ReLU
        for layer in self.hidden_layers:
            x = jax.nn.relu(layer(x))

        # Project to output size
        x = self.linear(x)

        # Denormalize output
        return x * self.y_std[...] + self.y_mean[...]
