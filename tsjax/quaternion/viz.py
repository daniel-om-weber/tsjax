"""Quaternion-specific visualization functions."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import matplotlib.pyplot as plt


def plot_quaternion_results(
    target: np.ndarray,
    pred: np.ndarray,
    n: int = 4,
    figsize: tuple[float, float] | None = None,
    u: np.ndarray | None = None,
    u_labels: list[str] | None = None,
    **kwargs,
) -> tuple[plt.Figure, np.ndarray]:
    """Plot quaternion inclination angle, angular error, and optional input signals.

    Layout: (2 + has_u) rows x n columns.

    - Row 0: Inclination angle of target and prediction (degrees).
    - Row 1: Angular error between prediction and target (degrees).
    - Row 2 (if *u* given): Input signal channels.

    When used as a ``Learner.plot_results_fn`` callback, the extra keyword
    arguments (``batch``, ``signal_names``, ``pipeline``) are used to
    auto-extract *u* and *u_labels* if they are not provided explicitly.

    Parameters
    ----------
    target : array ``(bs, seq_len, 4)`` -- ground truth quaternions ``[w, x, y, z]``.
    pred : array ``(bs, seq_len, 4)`` -- predicted quaternions.
    n : Number of samples to plot (clamped to batch size).
    figsize : Matplotlib figure size.
    u : Optional input array ``(bs, seq_len, n_u)`` for context row.
    u_labels : Channel names for input signals.
    **kwargs : Absorbed for Learner callback compatibility
        (``batch``, ``signal_names``, ``pipeline``).

    Returns
    -------
    ``(fig, axes)`` where axes has shape ``(n_rows, n)``.
    """
    import matplotlib.pyplot as plt

    from ._np import inclination_angle, inclination_angle_abs

    target = np.asarray(target)
    pred = np.asarray(pred)
    n = min(n, target.shape[0])

    # Auto-extract input from Learner callback kwargs
    pipeline = kwargs.get("pipeline")
    batch = kwargs.get("batch")
    if u is None and pipeline is not None and batch is not None:
        input_key = pipeline.input_keys[0]
        u = np.asarray(batch[input_key])
        if u_labels is None:
            signal_names = kwargs.get("signal_names", {})
            u_labels = signal_names.get(input_key)

    has_u = u is not None

    # Compute angles (radians -> degrees)
    targ_incl = np.degrees(inclination_angle_abs(target))
    pred_incl = np.degrees(inclination_angle_abs(pred))
    error = np.degrees(inclination_angle(pred, target))

    if has_u:
        u = np.asarray(u)
        n_u = u.shape[2]
        if u_labels is None:
            u_labels = [f"u[{i}]" for i in range(n_u)]

    n_rows = 2 + (1 if has_u else 0)
    if figsize is None:
        figsize = (3.5 * n, 2.5 * n_rows)

    fig, axes = plt.subplots(n_rows, n, figsize=figsize, squeeze=False)

    for col in range(n):
        # Row 0: inclination angles
        ax = axes[0, col]
        ax.plot(targ_incl[col], label="Actual")
        ax.plot(pred_incl[col], "--", label="Predicted")
        ax.set_title(f"Sample {col}")
        if col == 0:
            ax.set_ylabel("Inclination [\u00b0]")
            ax.legend(fontsize="small")

        # Row 1: angular error
        ax = axes[1, col]
        ax.plot(error[col])
        if col == 0:
            ax.set_ylabel("Error [\u00b0]")

        # Row 2: input signals (if present)
        if has_u:
            ax = axes[2, col]
            for ch in range(n_u):
                ax.plot(u[col, :, ch], label=u_labels[ch])
            if col == 0:
                ax.set_ylabel("Input")
                if n_u > 1:
                    ax.legend(fontsize="small")

        axes[-1, col].set_xlabel("Time step")

    fig.tight_layout()
    return fig, axes
