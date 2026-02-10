"""Integration tests for RNN/GRU models."""

import jax.numpy as jnp
import numpy as np
from flax import nnx


def test_gru_output_shape():
    """GRU forward pass should produce (batch, seq_len, output_size)."""
    from tsjax import RNN

    model = RNN(input_size=1, output_size=1, hidden_size=8, rnn_type="gru", rngs=nnx.Rngs(0))
    x = jnp.ones((2, 10, 1))
    out = model(x)
    assert out.shape == (2, 10, 1)


def test_lstm_output_shape():
    """LSTM variant should produce the same shape."""
    from tsjax import RNN

    model = RNN(input_size=2, output_size=3, hidden_size=8, rnn_type="lstm", rngs=nnx.Rngs(0))
    x = jnp.ones((2, 10, 2))
    out = model(x)
    assert out.shape == (2, 10, 3)


def test_multilayer_rnn():
    """Multi-layer RNN should produce correct shape without error."""
    from tsjax import RNN

    model = RNN(
        input_size=1, output_size=1, hidden_size=8, num_layers=2, rnn_type="gru", rngs=nnx.Rngs(0)
    )
    x = jnp.ones((2, 20, 1))
    out = model(x)
    assert out.shape == (2, 20, 1)


def test_model_output_is_finite():
    """Model output should be finite for reasonable input."""
    from tsjax import RNN

    model = RNN(input_size=1, output_size=1, hidden_size=8, rnn_type="gru", rngs=nnx.Rngs(0))
    x = jnp.ones((2, 20, 1)) * 0.5
    out = model(x)
    assert jnp.all(jnp.isfinite(out))


def test_model_with_norm_stats():
    """Model with real norm stats should apply normalization/denormalization."""
    from tsjax import RNN

    u_mean, u_std = np.array([1.0]), np.array([2.0])
    y_mean, y_std = np.array([3.0]), np.array([4.0])

    model = RNN(
        input_size=1,
        output_size=1,
        hidden_size=8,
        rnn_type="gru",
        u_mean=u_mean,
        u_std=u_std,
        y_mean=y_mean,
        y_std=y_std,
        rngs=nnx.Rngs(0),
    )
    x = jnp.ones((2, 20, 1))
    out = model(x)
    assert out.shape == (2, 20, 1)
    assert jnp.all(jnp.isfinite(out))


def test_invalid_rnn_type():
    """Unknown rnn_type should raise ValueError."""
    import pytest

    from tsjax import RNN

    with pytest.raises(ValueError, match="Unknown rnn_type"):
        RNN(input_size=1, output_size=1, hidden_size=8, rnn_type="transformer", rngs=nnx.Rngs(0))
