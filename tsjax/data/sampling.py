"""Weighted sampling for Grain data pipelines."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypeVar

import grain
import numpy as np

from .sources import WindowedSource

T = TypeVar("T")


class WeightedMapDataset(grain.MapDataset[T]):
    """Index-expanding transform: elements with higher weight appear more often.

    Materializes an expanded index array where element *i* appears
    ``max(1, round(w_i / w_min))`` times.  A downstream ``.shuffle()``
    randomizes the access order; over one epoch every expanded slot is
    visited exactly once, so element frequencies are exactly proportional
    to the weights (up to integer rounding).

    Usage with ``.pipe()``::

        ds = (
            grain.MapDataset.source(source)
            .pipe(WeightedMapDataset, weights=weights)
            .seed(seed)
            .shuffle()
            .repeat(None)
        )
    """

    _MUTATES_ELEMENT_SPEC = False

    def __init__(self, parent: grain.MapDataset[T], weights: np.ndarray | Sequence[float]):
        super().__init__(parent)
        w = np.asarray(weights, dtype=np.float64)
        if len(w) != len(parent):
            raise ValueError(f"weights length {len(w)} != parent length {len(parent)}")
        if np.any(w <= 0):
            raise ValueError("All weights must be positive")
        counts = np.maximum(1, np.round(w / w.min()).astype(int))
        self._cum_counts = np.cumsum(counts)
        self._length = int(self._cum_counts[-1])
        self._parent_length = len(parent)

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index):
        if isinstance(index, slice):
            return self.slice(index)
        with self._stats.record_self_time():
            epoch, index_in_epoch = divmod(index, self._length)
            parent_index = int(np.searchsorted(self._cum_counts, index_in_epoch, side="right"))
            parent_index += epoch * self._parent_length
        return self._parent[parent_index]

    def _getitems(self, indices: Sequence[int]):
        with self._stats.record_self_time(num_elements=len(indices)):
            idx = np.asarray(indices)
            epoch, index_in_epoch = np.divmod(idx, self._length)
            parent_indices = np.searchsorted(self._cum_counts, index_in_epoch, side="right")
            parent_indices = parent_indices + epoch * self._parent_length
        return self._parent._getitems(parent_indices.tolist())

    def __str__(self) -> str:
        return f"WeightedMapDataset(len={self._length}, parent_len={self._parent_length})"


def uniform_file_weights(source: WindowedSource) -> np.ndarray:
    """Per-window weights so that each file contributes equally.

    Each window in file *f* receives weight ``1 / n_windows_f``, so the
    total weight per file sums to 1 regardless of file length.

    Parameters
    ----------
    source : WindowedSource
        Must be in windowed mode (``win_sz`` is set).

    Returns
    -------
    np.ndarray
        Weight array of shape ``(len(source),)``.
    """
    if not source.cum_windows:
        # Full-file mode: one sample per file, all equal
        return np.ones(len(source), dtype=np.float64)
    cum = source.cum_windows
    n_windows = np.diff([0] + cum)
    return np.repeat(1.0 / n_windows, n_windows)
