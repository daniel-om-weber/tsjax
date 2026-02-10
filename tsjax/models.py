"""RNN model with internal normalization using Flax NNX."""

from __future__ import annotations

import jax.numpy as jnp
from flax import nnx


class Buffer(nnx.Variable):
    """Non-trainable variable excluded from gradients."""

    pass


class RNN(nnx.Module):
    """Multi-layer RNN with internal input normalization and output denormalization.

    Raw physical values in, raw physical values out.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int = 100,
        num_layers: int = 1,
        rnn_type: str = "gru",
        u_mean: jnp.ndarray | None = None,
        u_std: jnp.ndarray | None = None,
        y_mean: jnp.ndarray | None = None,
        y_std: jnp.ndarray | None = None,
        *,
        rngs: nnx.Rngs,
    ):
        # Store norm stats as Buffer (excluded from gradients)
        self.u_mean = Buffer(jnp.zeros(input_size) if u_mean is None else jnp.asarray(u_mean))
        self.u_std = Buffer(jnp.ones(input_size) if u_std is None else jnp.asarray(u_std))
        self.y_mean = Buffer(jnp.zeros(output_size) if y_mean is None else jnp.asarray(y_mean))
        self.y_std = Buffer(jnp.ones(output_size) if y_std is None else jnp.asarray(y_std))

        # Select cell type
        match rnn_type.lower():
            case "gru":
                CellType = nnx.GRUCell
            case "lstm":
                CellType = nnx.OptimizedLSTMCell
            case _:
                raise ValueError(f"Unknown rnn_type: {rnn_type!r}. Use 'gru' or 'lstm'.")

        # Build multi-layer RNN
        layers = []
        for i in range(num_layers):
            in_feat = input_size if i == 0 else hidden_size
            cell = CellType(in_features=in_feat, hidden_features=hidden_size, rngs=rngs)
            layers.append(nnx.RNN(cell))
        self.rnn_layers = nnx.List(layers)

        # Output projection
        self.linear = nnx.Linear(in_features=hidden_size, out_features=output_size, rngs=rngs)

    def __call__(self, x):
        """Forward pass: raw input -> raw output.

        x: (batch, seq_len, input_size) — raw physical values
        returns: (batch, seq_len, output_size) — raw physical values
        """
        # Normalize input
        x = (x - self.u_mean[...]) / self.u_std[...]

        # Multi-layer RNN
        for rnn in self.rnn_layers:
            x = rnn(x)

        # Project to output size
        x = self.linear(x)

        # Denormalize output
        return x * self.y_std[...] + self.y_mean[...]


GRU = RNN  # backward compat
