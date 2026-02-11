"""Compute normalization statistics from a DataLoader."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterable


@dataclass(frozen=True)
class NormStats:
    """Per-channel normalization statistics."""

    mean: np.ndarray  # (n_channels,)
    std: np.ndarray  # (n_channels,)
    min: np.ndarray  # (n_channels,)
    max: np.ndarray  # (n_channels,)


IDENTITY_STATS = NormStats(
    mean=np.zeros(1, dtype=np.float32),
    std=np.ones(1, dtype=np.float32),
    min=np.zeros(1, dtype=np.float32),
    max=np.zeros(1, dtype=np.float32),
)

EMPTY_STATS = NormStats(
    mean=np.array([], dtype=np.float32),
    std=np.array([], dtype=np.float32),
    min=np.array([], dtype=np.float32),
    max=np.array([], dtype=np.float32),
)


def compute_stats(
    dl: Iterable[dict[str, np.ndarray]],
    keys: list[str] | None = None,
    *,
    n_batches: int = 10,
) -> dict[str, NormStats]:
    """Compute normalization stats by sampling batches from a DataLoader.

    Parameters
    ----------
    dl : iterable yielding ``dict[str, ndarray]`` batches.
    keys : Batch keys to compute stats for.  ``None`` infers from the first batch.
    n_batches : Number of batches to sample.
    """
    collected: dict[str, list[np.ndarray]] = {}

    for i, batch in enumerate(dl):
        if i >= n_batches:
            break
        if keys is None:
            keys = list(batch.keys())
        for key in keys:
            collected.setdefault(key, []).append(np.asarray(batch[key]))

    if not collected:
        keys = keys or []
        return {k: EMPTY_STATS for k in keys}

    result: dict[str, NormStats] = {}
    for key, arrays in collected.items():
        cat = np.concatenate(arrays, axis=0)  # (N, ..., C)
        # Flatten all dims except last channel dim → (M, C)
        if cat.ndim > 2:
            cat = cat.reshape(-1, cat.shape[-1])
        elif cat.ndim == 1:
            cat = cat[:, np.newaxis]

        mean = cat.mean(axis=0).astype(np.float32)
        std = np.maximum(cat.std(axis=0), 1e-8).astype(np.float32)
        mn = cat.min(axis=0).astype(np.float32)
        mx = cat.max(axis=0).astype(np.float32)
        result[key] = NormStats(mean=mean, std=std, min=mn, max=mx)

    return result
