"""Integration tests for loss functions."""

import jax.numpy as jnp


def test_normalized_mse_zero_for_identical():
    """MSE of identical predictions and targets should be 0."""
    from tsjax import normalized_mse

    pred = jnp.ones((4, 50, 2))
    target = jnp.ones((4, 50, 2))
    y_mean = jnp.array([0.0, 0.0])
    y_std = jnp.array([1.0, 1.0])
    loss = normalized_mse(pred, target, y_mean, y_std)
    assert float(loss) == 0.0


def test_normalized_mae_zero_for_identical():
    """MAE of identical predictions and targets should be 0."""
    from tsjax import normalized_mae

    pred = jnp.ones((4, 50, 1))
    target = jnp.ones((4, 50, 1))
    loss = normalized_mae(pred, target, jnp.array([0.0]), jnp.array([1.0]))
    assert float(loss) == 0.0


def test_rmse_zero_for_identical():
    """RMSE of identical predictions and targets should be 0."""
    from tsjax import rmse

    pred = jnp.ones((4, 50, 1))
    target = jnp.ones((4, 50, 1))
    loss = rmse(pred, target, jnp.array([0.0]), jnp.array([1.0]))
    assert float(loss) == 0.0


def test_losses_positive_for_different():
    """All loss functions should return positive values for different inputs."""
    from tsjax import normalized_mae, normalized_mse, rmse

    pred = jnp.ones((4, 50, 1))
    target = jnp.zeros((4, 50, 1))
    y_mean = jnp.array([0.0])
    y_std = jnp.array([1.0])

    assert float(normalized_mse(pred, target, y_mean, y_std)) > 0
    assert float(normalized_mae(pred, target, y_mean, y_std)) > 0
    assert float(rmse(pred, target, y_mean, y_std)) > 0


def test_normalized_mse_scale_invariance():
    """Scaling pred and target by the same factor should not change normalized_mse."""
    from tsjax import normalized_mse

    pred = jnp.array([[[2.0], [3.0]]])
    target = jnp.array([[[1.0], [4.0]]])
    y_mean = jnp.array([0.0])
    y_std = jnp.array([1.0])

    loss_base = normalized_mse(pred, target, y_mean, y_std)

    # Scale everything by 10 — normalized loss should be the same
    scale = 10.0
    loss_scaled = normalized_mse(
        pred * scale,
        target * scale,
        y_mean * scale,
        jnp.array([y_std[0] * scale]),
    )
    assert abs(float(loss_base) - float(loss_scaled)) < 1e-5


# ---------------------------------------------------------------------------
# Cross-entropy loss tests (Phase 6)
# ---------------------------------------------------------------------------


def test_cross_entropy_correct_prediction():
    """Cross-entropy should be low when logits strongly favor the correct class."""
    from tsjax import cross_entropy_loss

    # 4 samples, 3 classes — logits strongly favor class 0, 1, 2, 0
    pred = jnp.array([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0], [10.0, 0.0, 0.0]])
    target = jnp.array([[0.0], [1.0], [2.0], [0.0]])  # float from pipeline
    y_mean = jnp.array([0.0])
    y_std = jnp.array([1.0])
    loss = cross_entropy_loss(pred, target, y_mean, y_std)
    assert float(loss) < 0.01


def test_cross_entropy_wrong_prediction():
    """Cross-entropy should be high when logits favor the wrong class."""
    from tsjax import cross_entropy_loss

    pred = jnp.array([[10.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    target = jnp.array([[2.0], [1.0]])
    loss = cross_entropy_loss(pred, target, jnp.zeros(1), jnp.ones(1))
    assert float(loss) > 5.0


def test_cross_entropy_positive():
    """Cross-entropy should always be non-negative."""
    from tsjax import cross_entropy_loss

    pred = jnp.array([[1.0, 2.0, 3.0]])
    target = jnp.array([[2.0]])
    loss = cross_entropy_loss(pred, target, jnp.zeros(1), jnp.ones(1))
    assert float(loss) >= 0.0


def test_cross_entropy_1d_target():
    """Cross-entropy should handle 1d target (batch,)."""
    from tsjax import cross_entropy_loss

    pred = jnp.array([[1.0, 2.0], [2.0, 1.0]])
    target = jnp.array([1, 0])  # already 1d
    loss = cross_entropy_loss(pred, target, jnp.zeros(1), jnp.ones(1))
    assert jnp.isfinite(loss)


def test_cross_entropy_ignores_stats():
    """Cross-entropy should produce same loss regardless of y_mean/y_std."""
    from tsjax import cross_entropy_loss

    pred = jnp.array([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])
    target = jnp.array([[2.0], [0.0]])
    loss_a = cross_entropy_loss(pred, target, jnp.zeros(1), jnp.ones(1))
    loss_b = cross_entropy_loss(pred, target, jnp.array([99.0]), jnp.array([0.5]))
    assert abs(float(loss_a) - float(loss_b)) < 1e-6
