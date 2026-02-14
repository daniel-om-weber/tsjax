"""MinGRU model — parallelizable GRU variant using associative scan."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx


def _log_g(x):
    return jnp.where(x >= 0, jnp.log(jax.nn.relu(x) + 0.5), -jax.nn.softplus(-x))


def _logcumsumexp(x, axis=-2):
    return jax.lax.associative_scan(jnp.logaddexp, x, axis=axis)


def _parallel_scan_log(log_coeffs, log_values):
    a_star = jnp.cumsum(log_coeffs, axis=-2)
    log_h0_plus_b_star = _logcumsumexp(log_values - a_star, axis=-2)
    log_h = a_star + log_h0_plus_b_star
    return jnp.exp(log_h)


class _MinGRULayer(nnx.Module):
    """Single minGRU layer: scan + residual + LayerNorm."""

    def __init__(self, d_model: int, *, rngs: nnx.Rngs):
        self.linear = nnx.Linear(in_features=d_model, out_features=d_model * 2, rngs=rngs)
        self.norm = nnx.LayerNorm(num_features=d_model, rngs=rngs)

    def __call__(self, x):
        projected = self.linear(x)  # (*batch, seq_len, d_model * 2)
        h_tilde, k = jnp.split(projected, 2, axis=-1)

        log_coeffs = -jax.nn.softplus(k)  # log(1 - z)
        log_z = -jax.nn.softplus(-k)  # log(z)
        log_values = log_z + _log_g(h_tilde)

        h = _parallel_scan_log(log_coeffs, log_values)
        return self.norm(x + h)


class MinGRU(nnx.Module):
    """Multi-layer minGRU with parallel log-space scan.

    A parallelizable GRU variant where gates depend only on the input,
    enabling O(log n) parallel computation via ``jax.lax.associative_scan``.
    Uses input projection, residual connections, and LayerNorm following
    the architecture patterns from the original paper.

    Wrap with :class:`NormalizedModel` or ``nnx.Sequential(Normalize, MinGRU, Denormalize)``
    to handle raw physical values.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int = 100,
        num_layers: int = 1,
        *,
        rngs: nnx.Rngs,
    ):
        self.input_proj = nnx.Linear(in_features=input_size, out_features=hidden_size, rngs=rngs)
        self.layers = nnx.List([_MinGRULayer(hidden_size, rngs=rngs) for _ in range(num_layers)])
        self.output_proj = nnx.Linear(in_features=hidden_size, out_features=output_size, rngs=rngs)

    def __call__(self, u):
        """Forward pass: ``(*batch, seq_len, input_size)`` -> ``(*batch, seq_len, output_size)``."""
        x = self.input_proj(u)
        for layer in self.layers:
            x = layer(x)
        return self.output_proj(x)
