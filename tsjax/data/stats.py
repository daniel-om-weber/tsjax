"""Compute normalization statistics from HDF5 files.

Must match tsfast/datasets/core.py:57-84 (extract_mean_std_from_hdffiles) exactly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import h5py
import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable

    from .store import SignalStore


@dataclass(frozen=True)
class NormStats:
    """Per-channel normalization statistics."""

    mean: np.ndarray  # (n_channels,)
    std: np.ndarray  # (n_channels,)


IDENTITY_STATS = NormStats(mean=np.zeros(1, dtype=np.float32), std=np.ones(1, dtype=np.float32))


def compute_norm_stats(
    lst_files: list[str],
    lst_signals: list[str],
) -> NormStats:
    """Calculate mean and std of signals from HDF5 files.

    Matches the exact accumulation logic of extract_mean_std_from_hdffiles:
    - float64 accumulation for sums and squares
    - counts += data.size OUTSIDE per-signal loop (uses last signal's size)
    - Final cast to float32
    """
    if len(lst_signals) == 0:
        return NormStats(
            mean=np.array([], dtype=np.float32),
            std=np.array([], dtype=np.float32),
        )

    sums = np.zeros(len(lst_signals))
    squares = np.zeros(len(lst_signals))
    counts = 0

    for file in lst_files:
        with h5py.File(file, "r") as f:
            for i, signal in enumerate(lst_signals):
                data = f[signal][:]
                if data.ndim > 1:
                    raise ValueError(
                        f"Each dataset in a file has to be 1d. {signal} is {data.ndim}."
                    )
                sums[i] += np.sum(data)
                squares[i] += np.sum(data**2)

        counts += data.size

    means = sums / counts
    stds = np.sqrt((squares / counts) - (means**2))

    return NormStats(mean=means.astype(np.float32), std=stds.astype(np.float32))


def compute_scalar_stats(
    paths: list[str],
    attr_names: list[str],
) -> NormStats:
    """Compute mean/std of per-file HDF5 attributes.

    Parameters
    ----------
    paths : HDF5 file paths to iterate.
    attr_names : Root-level attribute names to read from each file.
    """
    if len(attr_names) == 0:
        return NormStats(
            mean=np.array([], dtype=np.float32),
            std=np.array([], dtype=np.float32),
        )

    sums = np.zeros(len(attr_names))
    squares = np.zeros(len(attr_names))
    count = 0

    for path in paths:
        with h5py.File(path, "r") as f:
            for i, attr in enumerate(attr_names):
                val = float(f.attrs[attr])
                sums[i] += val
                squares[i] += val**2
        count += 1

    means = sums / count
    stds = np.sqrt((squares / count) - (means**2))
    # Prevent zero std (single-value or constant attrs)
    stds = np.maximum(stds, 1e-8)

    return NormStats(mean=means.astype(np.float32), std=stds.astype(np.float32))


def compute_norm_stats_from_index(
    index: SignalStore,
    signals: list[str],
) -> NormStats:
    """Calculate mean and std of signals via a :class:`SignalStore`.

    Works with any SignalStore implementation including
    :class:`~tsjax.data.resample.ResampledStore`, so that norm stats
    reflect resampled data.

    The accumulation logic intentionally mirrors :func:`compute_norm_stats`
    (``counts += data.size`` uses the last signal's size per file).
    """
    if len(signals) == 0:
        return NormStats(
            mean=np.array([], dtype=np.float32),
            std=np.array([], dtype=np.float32),
        )

    sums = np.zeros(len(signals))
    squares = np.zeros(len(signals))
    counts = 0

    for path in index.paths:
        for i, signal in enumerate(signals):
            seq_len = index.get_seq_len(path, signal)
            data = index.read_signals(path, [signal], 0, seq_len)[:, 0]
            sums[i] += np.sum(data)
            squares[i] += np.sum(data**2)

        counts += seq_len

    means = sums / counts
    stds = np.sqrt((squares / counts) - (means**2))

    return NormStats(mean=means.astype(np.float32), std=stds.astype(np.float32))


def compute_stats_with_transform(
    source,
    key: str,
    transform: Callable[[np.ndarray], np.ndarray],
) -> NormStats:
    """Compute normalization stats for a key after applying a transform.

    Iterates the entire *source* (typically the training ``ComposedSource``),
    applies *transform* to each sample's *key* value, and accumulates
    per-channel mean/std on the transformed output.
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
