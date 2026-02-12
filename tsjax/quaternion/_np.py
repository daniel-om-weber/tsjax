"""NumPy quaternion helpers for data pipeline (augmentation + SLERP).

Quaternion convention: [w, x, y, z] with w as the scalar component.
All functions support arbitrary leading dimensions via ``...`` broadcasting.
"""

from __future__ import annotations

import numpy as np


def multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product of two quaternions, ``(..., 4)``."""
    a, b, c, d = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    e, f, g, h = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    return np.stack(
        [
            a * e - b * f - c * g - d * h,
            a * f + b * e + c * h - d * g,
            a * g - b * h + c * e + d * f,
            a * h + b * g - c * f + d * e,
        ],
        axis=-1,
    )


def conjugate(q: np.ndarray) -> np.ndarray:
    """Quaternion conjugate: ``[w, -x, -y, -z]``."""
    return q * np.array([1.0, -1.0, -1.0, -1.0])


def normalize(q: np.ndarray) -> np.ndarray:
    """L2-normalize quaternion to unit length."""
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def relative(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Relative quaternion: ``q1 * inv(q2)``, direct formula."""
    a, b, c, d = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    e, f, g, h = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    return np.stack(
        [
            a * e + b * f + c * g + d * h,
            -a * f + b * e - c * h + d * g,
            -a * g + b * h + c * e - d * f,
            -a * h - b * g + c * f + d * e,
        ],
        axis=-1,
    )


def from_angle_axis(angle: np.ndarray, axis: np.ndarray) -> np.ndarray:
    """Create quaternion from angle and axis of rotation.

    Parameters
    ----------
    angle : ``(N,)`` rotation angles in radians.
    axis : ``(N, 3)`` rotation axes (will be normalized).
    """
    axis = axis / np.linalg.norm(axis, axis=-1, keepdims=True)
    half = angle[..., np.newaxis] / 2.0
    return np.concatenate([np.cos(half), axis * np.sin(half)], axis=-1)


def rot_vec(v: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Rotate 3D vector(s) by quaternion(s).

    Parameters
    ----------
    v : ``(..., 3)`` vectors.
    q : ``(..., 4)`` or ``(4,)`` unit quaternion.
    """
    v_quat = np.concatenate([np.zeros_like(v[..., :1]), v], axis=-1)
    return multiply(conjugate(q), multiply(v_quat, q))[..., 1:]


def rand_quat(rng: np.random.Generator) -> np.ndarray:
    """Random unit quaternion using NumPy RNG."""
    q = rng.uniform(-1, 1, size=4).astype(np.float32)
    q /= np.linalg.norm(q)
    return q
