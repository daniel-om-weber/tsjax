"""Grain data sources for time series pipelines."""

from __future__ import annotations

import bisect
from collections.abc import Callable
from dataclasses import dataclass

import h5py
import numpy as np

from .store import SignalStore

# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------

ReaderFn = Callable[[str, int, int], np.ndarray]
"""Callable reader: ``(path, l_slc, r_slc) -> ndarray``."""

# ---------------------------------------------------------------------------
# Internal picklable reader for list[str] signal specs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _StoreReader:
    """Read signals from a store for a given window (picklable for Grain)."""

    store: SignalStore
    signals: list[str]

    def __call__(self, path: str, l_slc: int, r_slc: int) -> np.ndarray:
        return self.store.read_signals(path, self.signals, l_slc, r_slc)


# ---------------------------------------------------------------------------
# Helper reader factories (return picklable callables)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScalarAttrs:
    """Pre-cached per-file scalar attributes -> ``(n_attrs,)``."""

    _cache: dict[str, np.ndarray]

    def __call__(self, path: str, l_slc: int, r_slc: int) -> np.ndarray:
        return self._cache[path]


def scalar_attrs(paths: list[str], attr_names: list[str]) -> ScalarAttrs:
    """Read HDF5 scalar attributes and return a picklable callable.

    Parameters
    ----------
    paths : list[str]
        HDF5 file paths to read attributes from.
    attr_names : list[str]
        Attribute names to read from each file's root group.

    Returns
    -------
    ScalarAttrs
        Callable ``(path, l_slc, r_slc) -> ndarray`` of shape ``(len(attr_names),)``.
    """
    cache: dict[str, np.ndarray] = {}
    for path in paths:
        with h5py.File(path, "r") as f:
            vals = [float(f.attrs[a]) for a in attr_names]
        cache[path] = np.array(vals, dtype=np.float32)
    return ScalarAttrs(cache)


@dataclass(frozen=True)
class SignalFeature:
    """Read windowed signals, apply reduction -> ``(n_features,)``."""

    store: SignalStore
    signals: list[str]
    fn: Callable

    def __call__(self, path: str, l_slc: int, r_slc: int) -> np.ndarray:
        data = self.store.read_signals(path, self.signals, l_slc, r_slc)
        return self.fn(data).astype(np.float32)


def signal_feature(store: SignalStore, signals: list[str], fn: Callable) -> SignalFeature:
    """Create a picklable callable that reads signals and applies a reduction.

    Parameters
    ----------
    store : SignalStore
        Store to read signals from.
    signals : list[str]
        Signal names to read.
    fn : callable
        Reduction function ``(win_sz, n_signals) -> (n_features,)``.

    Returns
    -------
    SignalFeature
        Callable ``(path, l_slc, r_slc) -> ndarray``.
    """
    return SignalFeature(store, signals, fn)


# ---------------------------------------------------------------------------
# Grain-compatible sources
# ---------------------------------------------------------------------------


def _normalize_signals(
    store: SignalStore,
    signals: dict[str, list[str] | ReaderFn],
) -> tuple[dict[str, ReaderFn], dict[str, list[str]], str | None]:
    """Convert a mixed signal dict into uniform callables.

    Returns ``(readers, signal_keys, ref_signal)`` where:

    - *readers*: all entries normalized to callables
    - *signal_keys*: key -> signal names for ``list[str]`` entries (for validation)
    - *ref_signal*: first signal from the first ``list[str]`` entry, or ``None``
    """
    readers: dict[str, ReaderFn] = {}
    signal_keys: dict[str, list[str]] = {}
    ref_signal: str | None = None
    for key, spec in signals.items():
        if isinstance(spec, list):
            signal_keys[key] = spec
            if ref_signal is None:
                ref_signal = spec[0]
            readers[key] = _StoreReader(store, spec)
        else:
            readers[key] = spec
    return readers, signal_keys, ref_signal


class WindowedSource:
    """Grain-compatible source for sliding-window time series iteration.

    Maps a global index to a (file, window_start, window_end) triple,
    then calls each reader to extract that window.

    Window formula matches tsfast/data/core.py:150:
        n_win = max(0, (seq_len - win_sz) // stp_sz + 1)
    """

    def __init__(
        self,
        store: SignalStore,
        signals: dict[str, list[str] | ReaderFn],
        *,
        win_sz: int,
        stp_sz: int = 1,
    ):
        self.store = store
        self.win_sz = win_sz
        self.stp_sz = stp_sz

        readers, signal_keys, ref_signal = _normalize_signals(store, signals)
        self._readers = readers
        self._signal_keys = signal_keys

        if ref_signal is None:
            raise ValueError("At least one list[str] signal entry is required")
        self.ref_signal = ref_signal

        # Build cumulative window index
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

        self._validate_lengths()

    def _validate_lengths(self):
        """Check that all signal-based entries agree on sequence lengths per file."""
        for path in self.file_paths:
            ref_len = self.store.get_seq_len(path, self.ref_signal)
            for key, sigs in self._signal_keys.items():
                for sig in sigs:
                    sig_len = self.store.get_seq_len(path, sig)
                    if sig_len != ref_len:
                        raise ValueError(
                            f"Signal length mismatch in {path!r}: "
                            f"ref signal {self.ref_signal!r} has {ref_len} samples, "
                            f"but entry {key!r} signal {sig!r} has {sig_len}. "
                            f"All windowed signals must have the same length."
                        )

    @property
    def signal_names(self) -> dict[str, list[str]]:
        """Signal names for ``list[str]`` entries (used for plot labels)."""
        return dict(self._signal_keys)

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, idx: int) -> dict[str, np.ndarray]:
        file_idx = bisect.bisect_right(self.cum_windows, idx)
        prev = self.cum_windows[file_idx - 1] if file_idx > 0 else 0
        local_win = idx - prev
        l_slc = local_win * self.stp_sz
        r_slc = l_slc + self.win_sz
        path = self.file_paths[file_idx]
        return {key: reader(path, l_slc, r_slc) for key, reader in self._readers.items()}


class FileSource:
    """Grain-compatible source for one-sample-per-file iteration.

    Each index maps to one file.  Readers receive ``(0, seq_len)`` bounds.
    """

    def __init__(
        self,
        store: SignalStore,
        signals: dict[str, list[str] | ReaderFn],
    ):
        self.store = store
        readers, signal_keys, ref_signal = _normalize_signals(store, signals)
        self._readers = readers
        self._signal_keys = signal_keys
        self.file_paths = list(store.paths)
        self._ref_signal = ref_signal
        self._seq_lens = self._compute_seq_lens()

    @property
    def signal_names(self) -> dict[str, list[str]]:
        """Signal names for ``list[str]`` entries (used for plot labels)."""
        return dict(self._signal_keys)

    def _compute_seq_lens(self) -> list[int]:
        if self._ref_signal is None:
            return [0] * len(self.file_paths)  # pure-scalar, seq_len unused
        return [self.store.get_seq_len(p, self._ref_signal) for p in self.file_paths]

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> dict[str, np.ndarray]:
        path = self.file_paths[idx]
        seq_len = self._seq_lens[idx]
        return {key: reader(path, 0, seq_len) for key, reader in self._readers.items()}


DataSource = WindowedSource | FileSource
"""Type alias for Grain-compatible time series sources."""
