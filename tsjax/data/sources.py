"""Grain data sources: composable Index + Reader architecture."""

from __future__ import annotations

import bisect
from collections.abc import Callable
from dataclasses import dataclass
from typing import ClassVar, Protocol, runtime_checkable

import h5py
import numpy as np

from .store import SignalStore

# ---------------------------------------------------------------------------
# Reader spec types — lightweight descriptors interpreted by the factory
# ---------------------------------------------------------------------------


@dataclass
class Signals:
    """Spec: read windowed signals -> (win_len, n_signals)."""

    signals: list[str]
    needs_windowing: ClassVar[bool] = True

    def build_reader(self, store, files):
        return SequenceReader(store, list(self.signals))


@dataclass
class ScalarAttr:
    """Spec: read per-file HDF5 attributes -> (n_attrs,)."""

    attrs: list[str]
    needs_windowing: ClassVar[bool] = False

    def build_reader(self, store, files):
        return ScalarAttrReader(files, self.attrs)


@dataclass
class Feature:
    """Spec: read windowed signals, apply reduction -> (n_features,)."""

    signals: list[str]
    fn: Callable  # (win_sz, n_ch) -> (n_features,)
    needs_windowing: ClassVar[bool] = True

    def build_reader(self, store, files):
        return FeatureReader(store, self.signals, self.fn)


ReaderSpec = list[str] | Signals | ScalarAttr | Feature

# ---------------------------------------------------------------------------
# Index classes — iteration strategy (private, used inside DataSource)
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
# Reader Protocol + concrete readers
# ---------------------------------------------------------------------------


@runtime_checkable
class Reader(Protocol):
    """Protocol for objects that read one data stream from a source."""

    signals: list[str]

    def __call__(self, path: str, l_slc: int, r_slc: int) -> np.ndarray: ...


class SequenceReader:
    """Read signals from a store — windowed slice or full sequence.

    When called with ``l_slc < r_slc``, reads the specified window.
    When called with ``l_slc >= r_slc`` (the _FileIndex convention),
    reads the full sequence.
    """

    def __init__(self, store: SignalStore, signals: list[str]):
        self.store = store
        self.signals = signals

    def __call__(self, path: str, l_slc: int, r_slc: int) -> np.ndarray:
        if l_slc >= r_slc:  # _FileIndex convention → full sequence
            r_slc = self.store.get_seq_len(path, self.signals[0])
            l_slc = 0
        return self.store.read_signals(path, self.signals, l_slc, r_slc)


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
        if l_slc >= r_slc:  # full sequence
            r_slc = self.store.get_seq_len(path, self.signals[0])
            l_slc = 0
        data = self.store.read_signals(path, self.signals, l_slc, r_slc)
        return self.fn(data).astype(np.float32)


# ---------------------------------------------------------------------------
# DataSource — Grain-compatible source combining an index with named readers
# ---------------------------------------------------------------------------


class DataSource:
    """Grain-compatible time series source.

    ``win_sz=None`` → full sequence per file (one sample per file).
    ``win_sz=int``  → sliding windows with ``stp_sz`` stride.

    Parameters
    ----------
    store : SignalStore used for index construction (paths, seq_len).
        Readers carry their own store references.
    readers : dict mapping batch key names to Reader objects.
    win_sz : Window size, or None for full-sequence mode.
    stp_sz : Step size for windowed iteration.
    """

    def __init__(
        self,
        store: SignalStore,
        readers: dict[str, Reader],
        *,
        win_sz: int | None = None,
        stp_sz: int = 1,
    ):
        self.store = store
        self.readers = dict(readers)
        if win_sz is not None:
            ref = self._find_ref_signal()
            self._index = _WindowIndex(store, win_sz, stp_sz, ref)
        else:
            self._index = _FileIndex(list(store.paths))
        self._validate_lengths()

    def _find_ref_signal(self) -> str:
        """Auto-detect reference signal from the first SequenceReader."""
        for reader in self.readers.values():
            if isinstance(reader, (SequenceReader, FeatureReader)):
                return reader.signals[0]
        raise ValueError("No SequenceReader or FeatureReader found — cannot determine ref_signal")

    def _validate_lengths(self):
        """Check that all sequence readers agree on sequence lengths per file.

        Compares every SequenceReader/FeatureReader against the index's
        ref_signal length for each file. Raises ValueError at construction
        time if any mismatch is found.
        """
        if not isinstance(self._index, _WindowIndex):
            return
        for path in self._index.store.paths:
            ref_len = self._index.store.get_seq_len(path, self._index.ref_signal)
            for key, reader in self.readers.items():
                if not isinstance(reader, (SequenceReader, FeatureReader)):
                    continue
                for sig in reader.signals:
                    sig_len = reader.store.get_seq_len(path, sig)
                    if sig_len != ref_len:
                        raise ValueError(
                            f"Signal length mismatch in {path!r}: "
                            f"ref signal {self._index.ref_signal!r} has {ref_len} samples, "
                            f"but reader {key!r} signal {sig!r} has {sig_len}. "
                            f"All windowed signals must have the same length."
                        )

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> dict[str, np.ndarray]:
        path, l_slc, r_slc = self._index.resolve(idx)
        return {key: reader(path, l_slc, r_slc) for key, reader in self.readers.items()}
