"""Compute normalization statistics from training data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable

    from .sources import DataSource


@dataclass(frozen=True)
class NormStats:
    """Per-channel normalization statistics."""

    mean: np.ndarray  # (n_channels,)
    std: np.ndarray  # (n_channels,)


IDENTITY_STATS = NormStats(mean=np.zeros(1, dtype=np.float32), std=np.ones(1, dtype=np.float32))

EMPTY_STATS = NormStats(
    mean=np.array([], dtype=np.float32),
    std=np.array([], dtype=np.float32),
)


def compute_stats(
    source: DataSource,
    key: str,
    *,
    transform: Callable[[np.ndarray], np.ndarray] | None = None,
) -> NormStats:
    """Compute normalization stats for a single key from a DataSource.

    If the reader for *key* has a ``compute_stats()`` method, it is called
    directly.  Otherwise a ``TypeError`` is raised advising the caller to
    pass pre-computed stats.

    When *transform* is given the reader method is bypassed: the function
    iterates all windowed samples, applies *transform*, and accumulates
    on the transformed output.

    Parameters
    ----------
    source : DataSource
        Typically the training-split source.
    key : str
        Batch key whose reader to use (e.g. ``"u"`` or ``"y"``).
    transform : callable, optional
        Per-sample transform applied before accumulation.  Takes an ndarray
        (the raw sample value for *key*) and returns an ndarray.
    """
    reader = source.readers.get(key)
    if reader is None:
        raise ValueError(f"No reader for key {key!r} in source")

    if transform is not None:
        return _transform_stats(source, key, transform)

    if hasattr(reader, "compute_stats"):
        return reader.compute_stats()

    raise TypeError(
        f"Cannot auto-compute stats for key {key!r} "
        f"(reader type {type(reader).__name__} has no compute_stats method). "
        f"Pass pre-computed stats instead."
    )


# ---------------------------------------------------------------------------
# Transform-based stats (reader-type-agnostic, operates on DataSource)
# ---------------------------------------------------------------------------


def _transform_stats(
    source: DataSource,
    key: str,
    transform: Callable[[np.ndarray], np.ndarray],
) -> NormStats:
    """Stats after applying a per-sample transform.

    Iterates the entire source (windowed samples), applies *transform*
    to each sample's *key* value, and accumulates per-channel mean/std.
    """
    sums = None
    squares = None
    count = 0

    for i in range(len(source)):
        sample = source[i]
        val = transform(sample[key]).astype(np.float32)

        if val.ndim <= 1:
            # Scalar output — one observation per sample
            n = val.shape[0] if val.ndim == 1 else 1
            if sums is None:
                sums = np.zeros(n, dtype=np.float64)
                squares = np.zeros(n, dtype=np.float64)
            sums += val
            squares += val**2
            count += 1
        else:
            # Sequence output: (seq_len, n_ch) — accumulate per-timestep
            if sums is None:
                sums = np.zeros(val.shape[-1], dtype=np.float64)
                squares = np.zeros(val.shape[-1], dtype=np.float64)
            sums += val.sum(axis=0)
            squares += (val**2).sum(axis=0)
            count += val.shape[0]

    means = sums / count
    stds = np.sqrt((squares / count) - (means**2))
    stds = np.maximum(stds, 1e-8)
    return NormStats(mean=means.astype(np.float32), std=stds.astype(np.float32))
