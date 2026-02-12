"""Quaternion data transforms: augmentation and SLERP interpolation."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from . import _np as qnp

# ---------------------------------------------------------------------------
# Quaternion augmentation (dict-level, modifies both input and target)
# ---------------------------------------------------------------------------


def quaternion_augmentation(
    inp_groups: list[tuple[int, int]],
    input_key: str = "u",
    target_key: str = "y",
    p: float = 1.0,
) -> Callable[[dict[str, np.ndarray], np.random.Generator], dict[str, np.ndarray]]:
    """Random rotation augmentation for quaternion/IMU data.

    Generates one random unit quaternion per sample and applies it to the
    specified channel groups in the input signal and to the target quaternion.

    Parameters
    ----------
    inp_groups : List of ``(start, end)`` channel index pairs (inclusive).
        3-channel groups are rotated as 3D vectors.
        4-channel groups are rotated via quaternion multiplication.
    input_key : Batch key for the input signal.
    target_key : Batch key for the target quaternion signal.
    p : Probability of applying the augmentation per sample.
    """
    for start, end in inp_groups:
        n_ch = end - start + 1
        if n_ch not in (3, 4):
            raise ValueError(f"Group ({start}, {end}) has {n_ch} channels, expected 3 or 4")

    def _augment(item: dict[str, np.ndarray], rng: np.random.Generator) -> dict[str, np.ndarray]:
        if p < 1.0 and rng.random() > p:
            return item
        q_rot = qnp.rand_quat(rng)
        u = item[input_key].copy()
        for start, end in inp_groups:
            slc = slice(start, end + 1)
            if end - start + 1 == 3:
                u[..., slc] = qnp.rot_vec(u[..., slc], q_rot)
            else:
                u[..., slc] = qnp.multiply(u[..., slc], q_rot)
        y = qnp.multiply(item[target_key], q_rot)
        return {**item, input_key: u, target_key: y}

    return _augment


# ---------------------------------------------------------------------------
# Quaternion SLERP interpolation
# ---------------------------------------------------------------------------


def quat_interp(
    quat: np.ndarray,
    ind: np.ndarray,
    extend: bool = False,
) -> np.ndarray:
    """SLERP quaternion interpolation at non-integer indices.

    Parameters
    ----------
    quat : ``(N, 4)`` input quaternions.
    ind : ``(M,)`` sampling indices in range ``[0, N-1]``.
    extend : If True, clamp to first/last quaternion for out-of-range
        indices. If False, fill with NaN.

    Returns
    -------
    ``(M, 4)`` interpolated quaternions, float32.
    """
    quat = quat.astype(np.float64)
    ind = np.atleast_1d(ind).astype(np.float64)
    N = quat.shape[0]

    ind0 = np.clip(np.floor(ind).astype(int), 0, N - 1)
    ind1 = np.clip(np.ceil(ind).astype(int), 0, N - 1)

    # Normalize to ensure unit quaternions (avoids precision issues)
    quat = quat / np.linalg.norm(quat, axis=-1, keepdims=True)

    q0 = quat[ind0]
    q1 = quat[ind1]
    # inv(q0) * q1: relative rotation from q0 to q1
    q_rel = qnp.multiply(qnp.conjugate(q0), q1)

    # Ensure positive w for shortest-path interpolation ([0, 180 deg])
    flip = q_rel[..., 0] < 0
    q_rel[flip] = -q_rel[flip]

    angle = 2 * np.arccos(np.clip(q_rel[..., 0], -1, 1))
    axis = q_rel[..., 1:]

    # For near-zero angles, copy the source quaternion directly
    direct = angle < 1e-3
    quat_out = np.empty_like(q0)
    quat_out[direct] = q0[direct]

    # Interpolate the rest via angle-axis
    interp = ~direct
    t01 = ind - ind0
    q_frac = qnp.from_angle_axis((t01 * angle)[interp], axis[interp])
    quat_out[interp] = qnp.multiply(q0[interp], q_frac)

    if not extend:
        quat_out[ind < 0] = np.nan
        quat_out[ind > N - 1] = np.nan

    return quat_out.astype(np.float32)
