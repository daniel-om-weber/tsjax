"""Grain data sources for time series pipelines."""

from __future__ import annotations

import bisect
import functools
from collections.abc import Callable
from dataclasses import dataclass

import h5py
import numpy as np

from .store import SignalStore

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
# Scalar attribute reader (picklable via functools.partial)
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=None)
def _read_hdf5_attrs(path: str, attr_names: tuple[str, ...]) -> np.ndarray:
    """Read scalar attributes from an HDF5 file (cached per path+attrs)."""
    with h5py.File(path, "r") as f:
        vals = [float(f.attrs[a]) for a in attr_names]
    return np.array(vals, dtype=np.float32)


def _scalar_attrs_reader(
    attr_names: tuple[str, ...], path: str, l_slc: int, r_slc: int
) -> np.ndarray:
    return _read_hdf5_attrs(path, attr_names)


def scalar_attrs(
    paths: list[str], attr_names: list[str]
) -> functools.partial[np.ndarray]:
    """Read HDF5 scalar attributes and return a picklable callable.

    Parameters
    ----------
    paths : list[str]
        HDF5 file paths to read attributes from.
    attr_names : list[str]
        Attribute names to read from each file's root group.

    Returns
    -------
    callable
        ``(path, l_slc, r_slc) -> ndarray`` of shape ``(len(attr_names),)``.
    """
    attr_tuple = tuple(attr_names)
    for path in paths:
        _read_hdf5_attrs(path, attr_tuple)  # pre-warm cache + validate
    return functools.partial(_scalar_attrs_reader, attr_tuple)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _normalize_signals(
    store: SignalStore,
    signals: dict[str, list[str] | Callable[[str, int, int], np.ndarray]],
) -> tuple[dict[str, Callable], dict[str, list[str]], str | None]:
    """Convert a mixed signal dict into uniform callables.

    Returns ``(readers, signal_keys, ref_signal)`` where:

    - *readers*: all entries normalized to callables
    - *signal_keys*: key -> signal names for ``list[str]`` entries (for validation)
    - *ref_signal*: first signal from the first ``list[str]`` entry, or ``None``
    """
    readers: dict[str, Callable] = {}
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


# ---------------------------------------------------------------------------
# Grain-compatible source
# ---------------------------------------------------------------------------


class WindowedSource:
    """Grain-compatible source for time series iteration.

    Maps a global index to a (file, window_start, window_end) triple,
    then calls each reader to extract that window.

    When ``win_sz=None``, operates in full-file mode: each index maps to
    one file and readers receive ``(0, seq_len)`` bounds.

    Window formula (when ``win_sz`` is set):
        n_win = max(0, (seq_len - win_sz) // stp_sz + 1)
    """

    def __init__(
        self,
        store: SignalStore,
        signals: dict[str, list[str] | Callable[[str, int, int], np.ndarray]],
        *,
        win_sz: int | None = None,
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

        self.file_paths: list[str] = list(store.paths)

        if win_sz is None:
            # Full-file mode: one sample per file
            self._len = len(self.file_paths)
            self._seq_lens = [store.get_seq_len(p, ref_signal) for p in self.file_paths]
            self.cum_windows: list[int] = []
        else:
            # Windowed mode: sliding windows over files
            self.cum_windows = []
            self._seq_lens = []
            total = 0
            for path in self.file_paths:
                seq_len = store.get_seq_len(path, ref_signal)
                n_win = max(0, (seq_len - win_sz) // stp_sz + 1)
                total += n_win
                self.cum_windows.append(total)
                self._seq_lens.append(seq_len)
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
        if self.win_sz is None:
            # Full-file mode
            path = self.file_paths[idx]
            seq_len = self._seq_lens[idx]
            return {key: reader(path, 0, seq_len) for key, reader in self._readers.items()}

        # Windowed mode
        file_idx = bisect.bisect_right(self.cum_windows, idx)
        prev = self.cum_windows[file_idx - 1] if file_idx > 0 else 0
        local_win = idx - prev
        l_slc = local_win * self.stp_sz
        r_slc = l_slc + self.win_sz
        path = self.file_paths[file_idx]
        return {key: reader(path, l_slc, r_slc) for key, reader in self._readers.items()}
