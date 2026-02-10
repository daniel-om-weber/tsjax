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
