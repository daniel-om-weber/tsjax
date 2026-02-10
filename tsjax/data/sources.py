"""Grain data sources for windowed and full-sequence signal reading."""

from __future__ import annotations

import bisect

import numpy as np

from .store import SignalStore


class WindowedSource:
    """Grain-compatible data source with computed windowing via bisect.

    Replaces CreateDict + DfHDFCreateWindows + HDF2Sequence.
    Window formula matches tsfast/data/core.py:150:
        n_win = max(0, (seq_len - win_sz) // stp_sz + 1)
    """

    def __init__(
        self,
        store: SignalStore,
        win_sz: int,
        stp_sz: int,
        input_signals: list[str],
        output_signals: list[str],
    ):
        self.store = store
        self.win_sz = win_sz
        self.stp_sz = stp_sz
        self.input_signals = input_signals
        self.output_signals = output_signals

        self.file_paths = []
        self.cum_windows = []
        total = 0
        for path in store.paths:
            seq_len = store.get_seq_len(path, input_signals[0])
            n_win = max(0, (seq_len - win_sz) // stp_sz + 1)
            total += n_win
            self.file_paths.append(path)
            self.cum_windows.append(total)
        self._len = total

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, idx: int) -> dict[str, np.ndarray]:
        file_idx = bisect.bisect_right(self.cum_windows, idx)
        prev = self.cum_windows[file_idx - 1] if file_idx > 0 else 0
        local_win = idx - prev
        l_slc = local_win * self.stp_sz
        r_slc = l_slc + self.win_sz
        path = self.file_paths[file_idx]

        u = self.store.read_signals(path, self.input_signals, l_slc, r_slc)
        y = self.store.read_signals(path, self.output_signals, l_slc, r_slc)
        return {"u": u, "y": y}


class FullSequenceSource:
    """One item per file, no windowing. For test split with bs=1."""

    def __init__(
        self,
        store: SignalStore,
        input_signals: list[str],
        output_signals: list[str],
    ):
        self.store = store
        self.input_signals = input_signals
        self.output_signals = output_signals
        self.file_paths = list(store.paths)

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> dict[str, np.ndarray]:
        path = self.file_paths[idx]
        seq_len = self.store.get_seq_len(path)
        u = self.store.read_signals(path, self.input_signals, 0, seq_len)
        y = self.store.read_signals(path, self.output_signals, 0, seq_len)
        return {"u": u, "y": y}
