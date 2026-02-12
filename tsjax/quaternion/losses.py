"""Quaternion losses and metrics for attitude estimation.

Loss and metric functions follow the tsjax Learner signature::

    fn(pred, target, y_mean, y_std) -> scalar

Quaternion losses ignore ``y_mean`` / ``y_std`` (quaternions are unit-scale).
"""

from __future__ import annotations

import functools

import jax.numpy as jnp

from ._algebra import inclination_angle, pitch_angle, quat_diff, relative_angle, roll_angle

# ---------------------------------------------------------------------------
# Loss functions — (pred, target, y_mean, y_std) -> scalar
# ---------------------------------------------------------------------------


def abs_inclination(pred, target, y_mean, y_std):
    """Mean absolute inclination angle (radians). Primary quaternion loss."""
    return jnp.abs(inclination_angle(pred, target)).mean()


def ms_inclination(pred, target, y_mean, y_std):
    """Mean squared inclination angle."""
    return (inclination_angle(pred, target) ** 2).mean()


def rms_inclination(pred, target, y_mean, y_std):
    """Root mean squared inclination angle."""
    return jnp.sqrt((inclination_angle(pred, target) ** 2).mean())


def smooth_inclination(pred, target, y_mean, y_std):
    """Smooth L1 (Huber) loss on inclination angle."""
    x = inclination_angle(pred, target)
    return jnp.where(jnp.abs(x) < 1.0, 0.5 * x**2, jnp.abs(x) - 0.5).mean()


def inclination_loss(pred, target, y_mean, y_std):
    """RMSE of ``sqrt(w**2 + z**2) - 1`` of the difference quaternion."""
    q = quat_diff(pred, target)
    err = jnp.sqrt(q[..., 0] ** 2 + q[..., 3] ** 2) - 1
    return jnp.sqrt((err**2).mean())


def inclination_loss_abs(pred, target, y_mean, y_std):
    """MAE of ``sqrt(w**2 + z**2) - 1`` of the difference quaternion."""
    q = quat_diff(pred, target)
    err = jnp.sqrt(q[..., 0] ** 2 + q[..., 3] ** 2) - 1
    return jnp.abs(err).mean()


def ms_rel_angle(pred, target, y_mean, y_std):
    """Mean squared relative (full rotation) angle."""
    return (relative_angle(pred, target) ** 2).mean()


def abs_rel_angle(pred, target, y_mean, y_std):
    """Mean absolute relative angle."""
    return jnp.abs(relative_angle(pred, target)).mean()


# ---------------------------------------------------------------------------
# Metrics — (pred, target, y_mean, y_std) -> scalar in degrees
# ---------------------------------------------------------------------------


def rms_inclination_deg(pred, target, y_mean, y_std):
    """RMS inclination angle in degrees."""
    return jnp.rad2deg(jnp.sqrt((inclination_angle(pred, target) ** 2).mean()))


def mean_inclination_deg(pred, target, y_mean, y_std):
    """Mean inclination angle in degrees."""
    return jnp.rad2deg(inclination_angle(pred, target).mean())


def rms_pitch_deg(pred, target, y_mean, y_std):
    """RMS pitch angle in degrees."""
    return jnp.rad2deg(jnp.sqrt((pitch_angle(pred, target) ** 2).mean()))


def rms_roll_deg(pred, target, y_mean, y_std):
    """RMS roll angle in degrees."""
    return jnp.rad2deg(jnp.sqrt((roll_angle(pred, target) ** 2).mean()))


def rms_rel_angle_deg(pred, target, y_mean, y_std):
    """RMS relative angle in degrees."""
    return jnp.rad2deg(jnp.sqrt((relative_angle(pred, target) ** 2).mean()))


def mean_rel_angle_deg(pred, target, y_mean, y_std):
    """Mean relative angle in degrees."""
    return jnp.rad2deg(relative_angle(pred, target).mean())


# ---------------------------------------------------------------------------
# NaN-safe wrapper
# ---------------------------------------------------------------------------


def nan_safe(fn):
    """Wrap a quaternion loss/metric to mask NaN target rows (JIT-compatible).

    Replaces NaN targets (and corresponding predictions) with the identity
    quaternion so they contribute zero error. The mean is corrected by the
    ratio ``n_total / n_valid`` to account for masked elements.

    The mask checks the last component of the target (``target[..., -1]``),
    matching TSFast's ``ignore_nan`` convention.
    """

    @functools.wraps(fn)
    def _wrapper(pred, target, y_mean, y_std):
        valid = ~jnp.isnan(target[..., -1])
        mask = valid[..., jnp.newaxis]
        identity = jnp.array([1.0, 0.0, 0.0, 0.0])
        pred_safe = jnp.where(mask, pred, identity)
        target_safe = jnp.where(mask, target, identity)
        n_total = jnp.array(valid.size, dtype=jnp.float32)
        n_valid = jnp.maximum(valid.sum(), 1).astype(jnp.float32)
        return fn(pred_safe, target_safe, y_mean, y_std) * (n_total / n_valid)

    return _wrapper
