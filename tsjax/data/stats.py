"""Compute normalization statistics from HDF5 files.

Must match tsfast/datasets/core.py:57-84 (extract_mean_std_from_hdffiles) exactly.
"""

from __future__ import annotations

import h5py
import numpy as np


def compute_norm_stats(
    lst_files: list[str],
    lst_signals: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate mean and std of signals from HDF5 files.

    Matches the exact accumulation logic of extract_mean_std_from_hdffiles:
    - float64 accumulation for sums and squares
    - counts += data.size OUTSIDE per-signal loop (uses last signal's size)
    - Final cast to float32
    """
    if len(lst_signals) == 0:
        return (None, None)

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

    return means.astype(np.float32), stds.astype(np.float32)
