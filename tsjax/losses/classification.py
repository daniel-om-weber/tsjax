"""Classification loss functions for tsjax models."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def cross_entropy_loss(pred, target, y_mean, y_std):
    """Cross-entropy loss on logits.  Ignores y_mean/y_std.

    Parameters
    ----------
    pred : (batch, n_classes) — raw logits.
    target : (batch, 1) or (batch,) — integer class labels (as float from pipeline).
    y_mean, y_std : Ignored (receives identity stats from pipeline).
    """
    if target.ndim > 1:
        target = target.squeeze(-1)
    target = target.astype(jnp.int32)
    log_probs = jax.nn.log_softmax(pred, axis=-1)
    return -jnp.mean(log_probs[jnp.arange(target.shape[0]), target])
