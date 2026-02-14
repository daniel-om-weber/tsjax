"""RNN model using Flax NNX."""

from __future__ import annotations

from flax import nnx


class RNN(nnx.Module):
    """Multi-layer RNN (normalized-space in, normalized-space out).

    Wrap with :class:`NormalizedModel` or ``nnx.Sequential(Normalize, RNN, Denormalize)``
    to handle raw physical values.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int = 100,
        num_layers: int = 1,
        cell_type: type = nnx.GRUCell,
        unroll: int = 1,
        *,
        rngs: nnx.Rngs,
    ):
        # Build multi-layer RNN
        layers = []
        for i in range(num_layers):
            in_feat = input_size if i == 0 else hidden_size
            cell = cell_type(in_features=in_feat, hidden_features=hidden_size, rngs=rngs)
            layers.append(nnx.RNN(cell, unroll=unroll))
        self.rnn_layers = nnx.List(layers)

        # Output projection
        self.linear = nnx.Linear(in_features=hidden_size, out_features=output_size, rngs=rngs)

    def __call__(self, u):
        """Forward pass: normalized input -> normalized output.

        u: (*batch, seq_len, input_size)
        returns: (*batch, seq_len, output_size)
        """
        x = u
        for rnn in self.rnn_layers:
            x = rnn(x)
        return self.linear(x)
