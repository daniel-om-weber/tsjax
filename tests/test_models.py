"""Integration tests for models and normalization layers."""

import jax.numpy as jnp
import numpy as np
from flax import nnx

# ---------------------------------------------------------------------------
# Normalize / Denormalize / NormalizedModel tests
# ---------------------------------------------------------------------------


def test_normalize_identity():
    """Normalize with default stats should be identity."""
    from tsjax import Normalize

    norm = Normalize(3)
    x = jnp.array([[1.0, 2.0, 3.0]])
    assert jnp.allclose(norm(x), x)


def test_normalize_with_stats():
    """Normalize should apply (x - mean) / std."""
    from tsjax import Normalize

    norm = Normalize(2, mean=np.array([1.0, 2.0]), std=np.array([2.0, 4.0]))
    x = jnp.array([[3.0, 6.0]])
    expected = jnp.array([[1.0, 1.0]])
    assert jnp.allclose(norm(x), expected)


def test_denormalize_identity():
    """Denormalize with default stats should be identity."""
    from tsjax import Denormalize

    denorm = Denormalize(3)
    x = jnp.array([[1.0, 2.0, 3.0]])
    assert jnp.allclose(denorm(x), x)


def test_denormalize_with_stats():
    """Denormalize should apply x * std + mean."""
    from tsjax import Denormalize

    denorm = Denormalize(2, mean=np.array([1.0, 2.0]), std=np.array([2.0, 4.0]))
    x = jnp.array([[1.0, 1.0]])
    expected = jnp.array([[3.0, 6.0]])
    assert jnp.allclose(denorm(x), expected)


def test_normalize_denormalize_roundtrip():
    """Normalize followed by Denormalize should recover original values."""
    from tsjax import Denormalize, Normalize

    mean, std = np.array([5.0, -3.0]), np.array([2.0, 0.5])
    norm = Normalize(2, mean=mean, std=std)
    denorm = Denormalize(2, mean=mean, std=std)
    x = jnp.array([[10.0, -1.0], [0.0, 3.0]])
    assert jnp.allclose(denorm(norm(x)), x, atol=1e-6)


def test_normalized_model_wraps_identity():
    """NormalizedModel with identity stats should not change model output."""
    from tsjax import MLP, Denormalize, Normalize, NormalizedModel

    model = MLP(input_size=3, output_size=2, hidden_sizes=[8], rngs=nnx.Rngs(0))
    wrapped = NormalizedModel(model, norm_in=Normalize(3), norm_out=Denormalize(2))
    x = jnp.ones((2, 3))
    assert jnp.allclose(wrapped(x), model(x))


def test_normalized_model_applies_stats():
    """NormalizedModel with real stats should differ from bare model."""
    from tsjax import MLP, Denormalize, Normalize, NormalizedModel

    model = MLP(input_size=2, output_size=1, hidden_sizes=[8], rngs=nnx.Rngs(0))
    wrapped = NormalizedModel(
        model,
        norm_in=Normalize(2, mean=np.array([1.0, 2.0]), std=np.array([3.0, 4.0])),
        norm_out=Denormalize(1, mean=np.array([5.0]), std=np.array([6.0])),
    )
    x = jnp.ones((2, 2))
    assert not jnp.allclose(wrapped(x), model(x), atol=1e-3)


# ---------------------------------------------------------------------------
# RNN tests
# ---------------------------------------------------------------------------


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


def test_model_with_norm_wrapper():
    """NormalizedModel-wrapped RNN should apply normalization/denormalization."""
    from tsjax import RNN, Denormalize, Normalize, NormalizedModel

    model = NormalizedModel(
        RNN(input_size=1, output_size=1, hidden_size=8, rnn_type="gru", rngs=nnx.Rngs(0)),
        norm_in=Normalize(1, mean=np.array([1.0]), std=np.array([2.0])),
        norm_out=Denormalize(1, mean=np.array([3.0]), std=np.array([4.0])),
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


# ---------------------------------------------------------------------------
# RNNEncoder tests
# ---------------------------------------------------------------------------


def test_encoder_gru_output_shape():
    """GRU encoder should produce (batch, output_size)."""
    from tsjax import RNNEncoder

    model = RNNEncoder(input_size=1, output_size=5, hidden_size=8, rnn_type="gru", rngs=nnx.Rngs(0))
    x = jnp.ones((2, 10, 1))
    out = model(x)
    assert out.shape == (2, 5)


def test_encoder_lstm_output_shape():
    """LSTM encoder should produce (batch, output_size)."""
    from tsjax import RNNEncoder

    model = RNNEncoder(
        input_size=3, output_size=7, hidden_size=16, rnn_type="lstm", rngs=nnx.Rngs(0)
    )
    x = jnp.ones((4, 20, 3))
    out = model(x)
    assert out.shape == (4, 7)


def test_encoder_multilayer():
    """Multi-layer encoder should produce correct shape."""
    from tsjax import RNNEncoder

    model = RNNEncoder(
        input_size=2, output_size=3, hidden_size=8, num_layers=2, rnn_type="gru", rngs=nnx.Rngs(0)
    )
    x = jnp.ones((2, 15, 2))
    out = model(x)
    assert out.shape == (2, 3)


def test_encoder_output_is_finite():
    """Encoder output should be finite for reasonable input."""
    from tsjax import RNNEncoder

    model = RNNEncoder(input_size=1, output_size=4, hidden_size=8, rnn_type="gru", rngs=nnx.Rngs(0))
    x = jnp.ones((2, 20, 1)) * 0.5
    out = model(x)
    assert jnp.all(jnp.isfinite(out))


def test_encoder_with_input_norm():
    """Encoder with Normalize wrapper should apply input normalization."""
    from tsjax import Denormalize, Normalize, NormalizedModel, RNNEncoder

    model = NormalizedModel(
        RNNEncoder(input_size=2, output_size=3, hidden_size=8, rnn_type="gru", rngs=nnx.Rngs(0)),
        norm_in=Normalize(2, mean=np.array([1.0, 2.0]), std=np.array([3.0, 4.0])),
        norm_out=Denormalize(3),  # identity
    )
    x = jnp.ones((2, 10, 2))
    out = model(x)
    assert out.shape == (2, 3)
    assert jnp.all(jnp.isfinite(out))


def test_encoder_output_denormalization():
    """Encoder with y stats should denormalize output (not raw logits)."""
    from tsjax import Denormalize, Normalize, NormalizedModel, RNNEncoder

    encoder = RNNEncoder(
        input_size=1, output_size=1, hidden_size=8, rnn_type="gru", rngs=nnx.Rngs(0)
    )
    model_denorm = NormalizedModel(
        encoder,
        norm_in=Normalize(1),
        norm_out=Denormalize(1, mean=np.array([10.0]), std=np.array([5.0])),
    )
    model_raw = NormalizedModel(
        RNNEncoder(input_size=1, output_size=1, hidden_size=8, rnn_type="gru", rngs=nnx.Rngs(0)),
        norm_in=Normalize(1),
        norm_out=Denormalize(1),
    )
    x = jnp.ones((2, 10, 1))
    out_denorm = model_denorm(x)
    out_raw = model_raw(x)
    assert out_denorm.shape == (2, 1)
    assert jnp.all(jnp.isfinite(out_denorm))
    assert not jnp.allclose(out_denorm, out_raw, atol=1e-3)


def test_encoder_invalid_rnn_type():
    """Unknown rnn_type should raise ValueError."""
    import pytest

    from tsjax import RNNEncoder

    with pytest.raises(ValueError, match="Unknown rnn_type"):
        RNNEncoder(
            input_size=1, output_size=2, hidden_size=8, rnn_type="transformer", rngs=nnx.Rngs(0)
        )


# ---------------------------------------------------------------------------
# MLP tests
# ---------------------------------------------------------------------------


def test_mlp_output_shape():
    """MLP should produce (batch, output_size)."""
    from tsjax import MLP

    model = MLP(input_size=3, output_size=2, hidden_sizes=[8, 4], rngs=nnx.Rngs(0))
    x = jnp.ones((4, 3))
    out = model(x)
    assert out.shape == (4, 2)


def test_mlp_default_hidden_sizes():
    """MLP with default hidden sizes should work."""
    from tsjax import MLP

    model = MLP(input_size=5, output_size=1, rngs=nnx.Rngs(0))
    x = jnp.ones((2, 5))
    out = model(x)
    assert out.shape == (2, 1)


def test_mlp_output_is_finite():
    """MLP output should be finite for reasonable input."""
    from tsjax import MLP

    model = MLP(input_size=3, output_size=2, hidden_sizes=[16], rngs=nnx.Rngs(0))
    x = jnp.ones((4, 3)) * 0.5
    out = model(x)
    assert jnp.all(jnp.isfinite(out))


def test_mlp_with_norm_wrapper():
    """MLP with NormalizedModel wrapper should apply normalization/denormalization."""
    from tsjax import MLP, Denormalize, Normalize, NormalizedModel

    model = NormalizedModel(
        MLP(input_size=2, output_size=1, hidden_sizes=[8], rngs=nnx.Rngs(0)),
        norm_in=Normalize(2, mean=np.array([1.0, 2.0]), std=np.array([3.0, 4.0])),
        norm_out=Denormalize(1, mean=np.array([5.0]), std=np.array([6.0])),
    )
    x = jnp.ones((2, 2))
    out = model(x)
    assert out.shape == (2, 1)
    assert jnp.all(jnp.isfinite(out))


def test_mlp_single_hidden():
    """MLP with a single hidden layer should produce correct shape."""
    from tsjax import MLP

    model = MLP(input_size=4, output_size=3, hidden_sizes=[16], rngs=nnx.Rngs(0))
    x = jnp.ones((2, 4))
    out = model(x)
    assert out.shape == (2, 3)
