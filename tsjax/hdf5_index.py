"""HDF5MmapIndex: picklable HDF5 reader using mmap for contiguous datasets."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import h5py


@dataclass(frozen=True)
class SignalInfo:
    offset: int | None  # None if chunked/compressed
    shape: tuple[int, ...]
    dtype_str: str
    is_contiguous: bool


class HDF5MmapIndex:
    """Extract mmap offsets from HDF5 files at init, then read via mmap at runtime.

    Fully picklable — stores only paths, offsets, shapes, and dtypes.
    Falls back to h5py for chunked/compressed datasets.
    """

    def __init__(self, hdf_paths, signal_names, preload=False):
        self.signal_names = list(signal_names)
        self.entries: dict[str, dict[str, SignalInfo]] = {}
        self._cache: dict[str, dict[str, np.ndarray]] | None = {} if preload else None
        for path in hdf_paths:
            path_str = str(path)
            with h5py.File(path_str, 'r') as f:
                self.entries[path_str] = {}
                if preload:
                    self._cache[path_str] = {}
                for name in signal_names:
                    ds = f[name]
                    layout = ds.id.get_create_plist().get_layout()
                    is_contiguous = layout == h5py.h5d.CONTIGUOUS
                    offset = ds.id.get_offset() if is_contiguous else None
                    self.entries[path_str][name] = SignalInfo(
                        offset=offset,
                        shape=tuple(ds.shape),
                        dtype_str=ds.dtype.str,
                        is_contiguous=is_contiguous,
                    )
                    if preload:
                        self._cache[path_str][name] = ds[:].astype(np.float32)

    def read_slice(self, path: str, signal: str, l_slc: int, r_slc: int) -> np.ndarray:
        """Read a window from a signal. Thread-safe."""
        path = str(path)
        if self._cache is not None:
            return self._cache[path][signal][l_slc:r_slc].copy()
        info = self.entries[path][signal]
        if info.is_contiguous:
            arr = np.memmap(
                path, dtype=info.dtype_str, mode='r',
                offset=info.offset, shape=info.shape,
            )
            return arr[l_slc:r_slc].copy()
        else:
            with h5py.File(path, 'r') as f:
                return f[signal][l_slc:r_slc]

    def read_signals(self, path: str, signals: list[str], l_slc: int, r_slc: int) -> np.ndarray:
        """Read and stack multiple signals into shape (window_len, n_signals)."""
        arrays = [self.read_slice(path, s, l_slc, r_slc) for s in signals]
        return np.stack(arrays, axis=-1).astype(np.float32)

    def get_seq_len(self, path: str, signal: str | None = None) -> int:
        """Get sequence length for a file. Uses first signal if none specified."""
        if signal is None:
            signal = self.signal_names[0]
        return self.entries[str(path)][signal].shape[0]
