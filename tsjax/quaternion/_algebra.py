"""Quaternion algebra and angle extraction (JAX).

Quaternion convention: ``[w, x, y, z]`` with *w* as the scalar component.
All functions support arbitrary leading dimensions via ``...`` broadcasting.
"""

from __future__ import annotations

import jax.numpy as jnp

# ---------------------------------------------------------------------------
# Core quaternion algebra
# ---------------------------------------------------------------------------


def quat_multiply(q1: jnp.ndarray, q2: jnp.ndarray) -> jnp.ndarray:
    """Hamilton product of two quaternions, ``(..., 4)``."""
    a, b, c, d = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    e, f, g, h = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    return jnp.stack(
        [
            a * e - b * f - c * g - d * h,
            a * f + b * e + c * h - d * g,
            a * g - b * h + c * e + d * f,
            a * h + b * g - c * f + d * e,
        ],
        axis=-1,
    )


def quat_conjugate(q: jnp.ndarray) -> jnp.ndarray:
    """Quaternion conjugate: ``[w, -x, -y, -z]``."""
    return q * jnp.array([1.0, -1.0, -1.0, -1.0])


def quat_normalize(q: jnp.ndarray) -> jnp.ndarray:
    """L2-normalize quaternion to unit length."""
    return q / jnp.linalg.norm(q, axis=-1, keepdims=True)


def quat_relative(q1: jnp.ndarray, q2: jnp.ndarray) -> jnp.ndarray:
    """Relative quaternion: ``q1 * inv(q2)``, direct formula."""
    a, b, c, d = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    e, f, g, h = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    return jnp.stack(
        [
            a * e + b * f + c * g + d * h,
            -a * f + b * e - c * h + d * g,
            -a * g + b * h + c * e - d * f,
            -a * h - b * g + c * f + d * e,
        ],
        axis=-1,
    )


def quat_diff(q1: jnp.ndarray, q2: jnp.ndarray, normalize: bool = True) -> jnp.ndarray:
    """Difference quaternion: optionally normalize, then compute relative."""
    if normalize:
        q1 = quat_normalize(q1)
        q2 = quat_normalize(q2)
    return quat_relative(q1, q2)


def rot_vec(v: jnp.ndarray, q: jnp.ndarray) -> jnp.ndarray:
    """Rotate 3D vector(s) ``(..., 3)`` by quaternion(s) ``(..., 4)``."""
    v_quat = jnp.concatenate([jnp.zeros_like(v[..., :1]), v], axis=-1)
    return quat_multiply(quat_conjugate(q), quat_multiply(v_quat, q))[..., 1:]


# ---------------------------------------------------------------------------
# Angle extraction (radians)
# ---------------------------------------------------------------------------


def inclination_angle(q1: jnp.ndarray, q2: jnp.ndarray) -> jnp.ndarray:
    """Inclination (tilt) angle between two quaternions.

    ``2 * atan2(sqrt(x**2 + y**2), sqrt(w**2 + z**2))`` of the difference
    quaternion.  Uses ``atan2`` instead of ``acos`` for numerical stability.

    Returns ``(...)`` angle in radians.
    """
    q = quat_diff(q1, q2)
    return 2 * jnp.arctan2(
        jnp.sqrt(q[..., 1] ** 2 + q[..., 2] ** 2),
        jnp.sqrt(q[..., 0] ** 2 + q[..., 3] ** 2),
    )


def relative_angle(q1: jnp.ndarray, q2: jnp.ndarray) -> jnp.ndarray:
    """Full rotation angle between two quaternions.

    ``2 * atan2(norm(xyz), |w|)`` of the difference quaternion.
    Uses ``atan2`` instead of ``acos`` for numerical stability.

    Returns ``(...)`` angle in radians.
    """
    q = quat_diff(q1, q2)
    return 2 * jnp.arctan2(jnp.linalg.norm(q[..., 1:], axis=-1), jnp.abs(q[..., 0]))


def roll_angle(q1: jnp.ndarray, q2: jnp.ndarray) -> jnp.ndarray:
    """Euler roll angle of the difference quaternion (radians)."""
    q = quat_diff(q1, q2)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return jnp.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))


def pitch_angle(q1: jnp.ndarray, q2: jnp.ndarray) -> jnp.ndarray:
    """Euler pitch angle of the difference quaternion (radians).

    Uses ``atan2(sin, cos)`` instead of ``asin`` for numerical stability
    near gimbal lock (``+/-pi/2``).
    """
    q = quat_diff(q1, q2)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    sin_p = jnp.clip(2.0 * (w * y - z * x), -1.0, 1.0)
    cos_p = jnp.sqrt(1.0 - sin_p**2)
    return jnp.arctan2(sin_p, cos_p)
