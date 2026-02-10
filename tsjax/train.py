"""Loss functions for tsjax models."""
from __future__ import annotations

import jax.numpy as jnp


def normalized_mse(pred, target, y_mean, y_std):
    """MSE in per-channel normalized space for balanced gradients."""
    pred_n = (pred - y_mean) / y_std
    targ_n = (target - y_mean) / y_std
    return jnp.mean((pred_n - targ_n) ** 2)


def normalized_mae(pred, target, y_mean, y_std):
    """MAE in per-channel normalized space for balanced gradients."""
    pred_n = (pred - y_mean) / y_std
    targ_n = (target - y_mean) / y_std
    return jnp.mean(jnp.abs(pred_n - targ_n))


def rmse(pred, target, y_mean, y_std):
    """Raw RMSE (comparable to TSFast's fun_rmse)."""
    return jnp.sqrt(jnp.mean((pred - target) ** 2))
