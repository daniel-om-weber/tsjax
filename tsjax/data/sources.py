"""Grain data sources: composable Index + Reader architecture."""

from __future__ import annotations

import bisect
from collections.abc import Callable
from dataclasses import dataclass

import h5py
import numpy as np

from .store import SignalStore

# ---------------------------------------------------------------------------
# Reader spec types — lightweight descriptors interpreted by the factory
# ---------------------------------------------------------------------------


@dataclass
class ScalarAttr:
    """Spec: read per-file HDF5 attributes -> (n_attrs,)."""

    attrs: list[str]


@dataclass
class Feature:
    """Spec: read windowed signals, apply reduction -> (n_features,)."""

    signals: list[str]
    fn: Callable  # (win_sz, n_ch) -> (n_features,)


ReaderSpec = list[str] | ScalarAttr | Feature

# ---------------------------------------------------------------------------
# Index classes — iteration strategy
# ---------------------------------------------------------------------------


class _WindowIndex:
    """Maps global index -> (path, l_slc, r_slc).  Windowed iteration.

    Window formula matches tsfast/data/core.py:150:
        n_win = max(0, (seq_len - win_sz) // stp_sz + 1)
    """

    def __init__(self, store: SignalStore, win_sz: int, stp_sz: int, ref_signal: str):
        self.store = store
        self.win_sz = win_sz
        self.stp_sz = stp_sz
        self.ref_signal = ref_signal

        self.file_paths: list[str] = []
        self.cum_windows: list[int] = []
        total = 0
        for path in store.paths:
            seq_len = store.get_seq_len(path, ref_signal)
            n_win = max(0, (seq_len - win_sz) // stp_sz + 1)
            total += n_win
            self.file_paths.append(path)
            self.cum_windows.append(total)
        self._len = total

    def __len__(self) -> int:
        return self._len

    def resolve(self, idx: int) -> tuple[str, int, int]:
        file_idx = bisect.bisect_right(self.cum_windows, idx)
        prev = self.cum_windows[file_idx - 1] if file_idx > 0 else 0
        local_win = idx - prev
        l_slc = local_win * self.stp_sz
        r_slc = l_slc + self.win_sz
        path = self.file_paths[file_idx]
        return path, l_slc, r_slc


class _FileIndex:
    """One sample per file — for test split or tabular data."""

    def __init__(self, paths: list[str]):
        self.file_paths = list(paths)

    def __len__(self) -> int:
        return len(self.file_paths)

    def resolve(self, idx: int) -> tuple[str, int, int]:
        return self.file_paths[idx], 0, 0


# ---------------------------------------------------------------------------
# Reader classes — how to read one data stream
# ---------------------------------------------------------------------------


class WindowedReader:
    """Read a slice of signals -> (win_sz, n_ch)."""

    def __init__(self, store: SignalStore, signals: list[str]):
        self.store = store
        self.signals = signals

    def __call__(self, path: str, l_slc: int, r_slc: int) -> np.ndarray:
        return self.store.read_signals(path, self.signals, l_slc, r_slc)


class FullSeqReader:
    """Read entire sequence -> (seq_len, n_ch).  Ignores l_slc/r_slc."""

    def __init__(self, store: SignalStore, signals: list[str]):
        self.store = store
        self.signals = signals

    def __call__(self, path: str, l_slc: int, r_slc: int) -> np.ndarray:
        seq_len = self.store.get_seq_len(path, self.signals[0])
        return self.store.read_signals(path, self.signals, 0, seq_len)


class ScalarAttrReader:
    """Read per-file HDF5 attributes -> (n_attrs,).  Pre-caches at init."""

    def __init__(self, paths: list[str], attr_names: list[str]):
        self.signals = list(attr_names)
        self._cache: dict[str, np.ndarray] = {}
        for path in paths:
            with h5py.File(path, "r") as f:
                vals = [float(f.attrs[a]) for a in attr_names]
            self._cache[path] = np.array(vals, dtype=np.float32)

    def __call__(self, path: str, l_slc: int, r_slc: int) -> np.ndarray:
        return self._cache[path]


class FeatureReader:
    """Read a window, apply a function -> (n_features,)."""

    def __init__(self, store: SignalStore, signals: list[str], fn: Callable):
        self.store = store
        self.signals = signals
        self.fn = fn

    def __call__(self, path: str, l_slc: int, r_slc: int) -> np.ndarray:
        data = self.store.read_signals(path, self.signals, l_slc, r_slc)
        return self.fn(data).astype(np.float32)


# ---------------------------------------------------------------------------
# Composed source
# ---------------------------------------------------------------------------


class ComposedSource:
    """Grain-compatible source combining an index with named readers."""

    def __init__(
        self,
        index: _WindowIndex | _FileIndex,
        readers: dict[str, WindowedReader | FullSeqReader | ScalarAttrReader | FeatureReader],
    ):
        self.index = index
        self.readers = readers
        self._validate_lengths()

    def _validate_lengths(self):
        """Check that all windowed readers agree on sequence lengths per file.

        Compares every WindowedReader/FullSeqReader against the index's
        ref_signal length for each file. Raises ValueError at construction
        time if any mismatch is found.
        """
        if not isinstance(self.index, _WindowIndex):
            return
        for path in self.index.store.paths:
            ref_len = self.index.store.get_seq_len(path, self.index.ref_signal)
            for key, reader in self.readers.items():
                if not isinstance(reader, (WindowedReader, FullSeqReader)):
                    continue
                for sig in reader.signals:
                    sig_len = reader.store.get_seq_len(path, sig)
                    if sig_len != ref_len:
                        raise ValueError(
                            f"Signal length mismatch in {path!r}: "
                            f"ref signal {self.index.ref_signal!r} has {ref_len} samples, "
                            f"but reader {key!r} signal {sig!r} has {sig_len}. "
                            f"All windowed signals must have the same length."
                        )

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> dict[str, np.ndarray]:
        path, l_slc, r_slc = self.index.resolve(idx)
        return {key: reader(path, l_slc, r_slc) for key, reader in self.readers.items()}
