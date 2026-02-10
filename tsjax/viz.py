"""Visualization functions for time series batches and model predictions."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import matplotlib.pyplot as plt


def plot_batch(
    batch: dict[str, np.ndarray],
    n: int = 4,
    figsize: tuple[float, float] | None = None,
    u_labels: list[str] | None = None,
    y_labels: list[str] | None = None,
) -> tuple[plt.Figure, np.ndarray]:
    """Plot input and output signals from a batch.

    Layout: 2 rows (input on top, output below) x n columns (one per sample).

    Parameters
    ----------
    batch : dict with "u" and "y" arrays of shape (bs, seq_len, n_channels).
    n : Number of samples to plot (clamped to batch size).
    figsize : Matplotlib figure size.
    u_labels : Channel names for input signals.
    y_labels : Channel names for output signals.

    Returns
    -------
    (fig, axes) where axes has shape (2, n).
    """
    import matplotlib.pyplot as plt

    u = np.asarray(batch["u"])
    y = np.asarray(batch["y"])
    n = min(n, u.shape[0])
    n_u, n_y = u.shape[2], y.shape[2]

    if u_labels is None:
        u_labels = [f"u[{i}]" for i in range(n_u)]
    if y_labels is None:
        y_labels = [f"y[{i}]" for i in range(n_y)]
    if figsize is None:
        figsize = (3.5 * n, 5)

    fig, axes = plt.subplots(2, n, figsize=figsize, squeeze=False)

    for col in range(n):
        ax_u = axes[0, col]
        for ch in range(n_u):
            ax_u.plot(u[col, :, ch], label=u_labels[ch])
        if n_u > 1:
            ax_u.legend(fontsize="small")
        ax_u.set_title(f"Sample {col}")

        ax_y = axes[1, col]
        for ch in range(n_y):
            ax_y.plot(y[col, :, ch], label=y_labels[ch])
        if n_y > 1:
            ax_y.legend(fontsize="small")

        axes[1, col].set_xlabel("Time step")

    axes[0, 0].set_ylabel("Input")
    axes[1, 0].set_ylabel("Output")
    fig.tight_layout()
    return fig, axes


def plot_results(
    target: np.ndarray,
    pred: np.ndarray,
    n: int = 4,
    figsize: tuple[float, float] | None = None,
    u: np.ndarray | None = None,
    y_labels: list[str] | None = None,
    u_labels: list[str] | None = None,
) -> tuple[plt.Figure, np.ndarray]:
    """Overlay actual vs predicted output signals.

    Layout: (has_u + n_y) rows x n columns. Input context on top row (if provided),
    then one row per output channel with actual vs predicted overlaid.

    Parameters
    ----------
    target : array (bs, seq_len, n_y) -- ground truth in raw space.
    pred : array (bs, seq_len, n_y) -- model predictions in raw space.
    n : Number of samples to plot (clamped to batch size).
    figsize : Matplotlib figure size.
    u : Optional input array (bs, seq_len, n_u) for context row.
    y_labels : Channel names for output signals.
    u_labels : Channel names for input signals.

    Returns
    -------
    (fig, axes) where axes has shape (n_rows, n).
    """
    import matplotlib.pyplot as plt

    target = np.asarray(target)
    pred = np.asarray(pred)
    n = min(n, target.shape[0])
    n_y = target.shape[2]
    has_u = u is not None

    if has_u:
        u = np.asarray(u)
        n_u = u.shape[2]
        if u_labels is None:
            u_labels = [f"u[{i}]" for i in range(n_u)]
    if y_labels is None:
        y_labels = [f"y[{i}]" for i in range(n_y)]

    n_rows = n_y + (1 if has_u else 0)
    if figsize is None:
        figsize = (3.5 * n, 2.5 * n_rows)

    fig, axes = plt.subplots(n_rows, n, figsize=figsize, squeeze=False)

    for col in range(n):
        row = 0

        # Optional input context row
        if has_u:
            ax_u = axes[row, col]
            for ch in range(n_u):
                ax_u.plot(u[col, :, ch], label=u_labels[ch])
            if n_u > 1:
                ax_u.legend(fontsize="small")
            if col == 0:
                ax_u.set_ylabel("Input")
            ax_u.set_title(f"Sample {col}")
            row += 1

        # Output rows: actual vs predicted
        for ch in range(n_y):
            ax = axes[row + ch, col]
            ax.plot(target[col, :, ch], label="Actual")
            ax.plot(pred[col, :, ch], "--", label="Predicted")
            if col == 0:
                label = y_labels[ch] if n_y > 1 else "Output"
                ax.set_ylabel(label)
                ax.legend(fontsize="small")
            if not has_u:
                ax.set_title(f"Sample {col}")

        axes[-1, col].set_xlabel("Time step")

    fig.tight_layout()
    return fig, axes


def plot_scalar_batch(
    batch: dict[str, np.ndarray],
    n: int = 4,
    figsize: tuple[float, float] | None = None,
    u_labels: list[str] | None = None,
    y_labels: list[str] | None = None,
) -> tuple[plt.Figure, np.ndarray]:
    """Plot input sequences with scalar targets annotated.

    For classification/regression where u is (bs, seq_len, n_ch) and y is
    (bs, n_attrs).

    Parameters
    ----------
    batch : dict with "u" (sequence) and "y" (scalar) arrays.
    n : Number of samples to plot.
    u_labels : Channel names for input signals.
    y_labels : Names for scalar target attributes.
    """
    import matplotlib.pyplot as plt

    u = np.asarray(batch["u"])
    y = np.asarray(batch["y"])
    n = min(n, u.shape[0])
    n_u = u.shape[2]

    if u_labels is None:
        u_labels = [f"u[{i}]" for i in range(n_u)]
    if y_labels is None:
        n_y = y.shape[-1] if y.ndim > 1 else 1
        y_labels = [f"y[{i}]" for i in range(n_y)]
    if figsize is None:
        figsize = (3.5 * n, 3)

    fig, axes = plt.subplots(1, n, figsize=figsize, squeeze=False)

    for col in range(n):
        ax = axes[0, col]
        for ch in range(n_u):
            ax.plot(u[col, :, ch], label=u_labels[ch])
        if n_u > 1:
            ax.legend(fontsize="small")

        # Annotate scalar target in title
        if y.ndim == 1:
            label = f"{y_labels[0]}={y[col]:.3g}"
        else:
            parts = [f"{y_labels[i]}={y[col, i]:.3g}" for i in range(y.shape[-1])]
            label = ", ".join(parts)
        ax.set_title(label)
        ax.set_xlabel("Time step")

    axes[0, 0].set_ylabel("Input")
    fig.tight_layout()
    return fig, axes


def plot_classification_results(
    target: np.ndarray,
    pred: np.ndarray,
    n: int = 4,
    figsize: tuple[float, float] | None = None,
    class_names: list[str] | None = None,
) -> tuple[plt.Figure, np.ndarray]:
    """Plot per-sample predicted class probabilities with true label.

    Parameters
    ----------
    target : (batch,) or (batch, 1) — integer class labels.
    pred : (batch, n_classes) — raw logits.
    n : Number of samples to plot.
    class_names : Names for each class.
    """
    import matplotlib.pyplot as plt

    target = np.asarray(target)
    pred = np.asarray(pred)
    if target.ndim > 1:
        target = target.squeeze(-1)
    target = target.astype(int)
    n = min(n, target.shape[0])
    n_classes = pred.shape[-1]

    # Softmax to get probabilities
    exp_pred = np.exp(pred - pred.max(axis=-1, keepdims=True))
    probs = exp_pred / exp_pred.sum(axis=-1, keepdims=True)

    if class_names is None:
        class_names = [str(i) for i in range(n_classes)]
    if figsize is None:
        figsize = (3.5 * n, 3)

    fig, axes = plt.subplots(1, n, figsize=figsize, squeeze=False)
    x_pos = np.arange(n_classes)

    for col in range(n):
        ax = axes[0, col]
        colors = [
            "#4CAF50" if i == target[col] else "#90CAF9" for i in range(n_classes)
        ]
        ax.bar(x_pos, probs[col], color=colors)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(class_names, fontsize="small")
        pred_cls = int(np.argmax(probs[col]))
        ax.set_title(
            f"True: {class_names[target[col]]}, Pred: {class_names[pred_cls]}"
        )
        ax.set_ylim(0, 1)

    axes[0, 0].set_ylabel("Probability")
    fig.tight_layout()
    return fig, axes


def plot_regression_scatter(
    target: np.ndarray,
    pred: np.ndarray,
    n: int | None = None,
    figsize: tuple[float, float] | None = None,
    y_labels: list[str] | None = None,
) -> tuple[plt.Figure, np.ndarray]:
    """Scatter plot of predicted vs actual values with identity diagonal.

    Parameters
    ----------
    target : (batch, n_outputs) or (batch,) — ground truth.
    pred : (batch, n_outputs) or (batch,) — predictions.
    n : Max samples to plot (None = all).
    y_labels : Names for output dimensions.
    """
    import matplotlib.pyplot as plt

    target = np.asarray(target)
    pred = np.asarray(pred)
    if target.ndim == 1:
        target = target[:, np.newaxis]
        pred = pred[:, np.newaxis]
    if n is not None:
        target = target[:n]
        pred = pred[:n]
    n_out = target.shape[-1]

    if y_labels is None:
        y_labels = [f"y[{i}]" for i in range(n_out)]
    if figsize is None:
        figsize = (4 * n_out, 4)

    fig, axes = plt.subplots(1, n_out, figsize=figsize, squeeze=False)

    for i in range(n_out):
        ax = axes[0, i]
        ax.scatter(target[:, i], pred[:, i], alpha=0.6, s=20)
        lo = min(target[:, i].min(), pred[:, i].min())
        hi = max(target[:, i].max(), pred[:, i].max())
        margin = (hi - lo) * 0.05 or 0.1
        ax.plot(
            [lo - margin, hi + margin], [lo - margin, hi + margin], "k--", alpha=0.4
        )
        ax.set_xlabel(f"Actual ({y_labels[i]})")
        ax.set_ylabel(f"Predicted ({y_labels[i]})")
        ax.set_title(y_labels[i])
        ax.set_aspect("equal", adjustable="datalim")

    fig.tight_layout()
    return fig, axes
