"""HDF5Store: picklable HDF5 reader using mmap for contiguous datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np


@dataclass(frozen=True)
class SignalInfo:
    offset: int | None  # None if chunked/compressed
    shape: tuple[int, ...]
    dtype_str: str
    is_contiguous: bool


class HDF5Store:
    """Extract mmap offsets from HDF5 files at init, then read via mmap at runtime.

    Fully picklable — stores only paths, offsets, shapes, and dtypes.
    Mmap objects are cached lazily and excluded from pickle state, so they
    are recreated automatically in grain worker processes (which use spawn).
    Falls back to h5py for chunked/compressed datasets.
    """

    def __init__(self, hdf_paths, signal_names, preload=False, dtype=np.float32):
        self.signal_names = list(signal_names)
        self.dtype = dtype
        self.entries: dict[str, dict[str, SignalInfo]] = {}
        self._cache: dict[str, dict[str, np.ndarray]] | None = {} if preload else None
        for path in hdf_paths:
            path_str = str(path)
            with h5py.File(path_str, "r") as f:
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
                        self._cache[path_str][name] = ds[:].astype(self.dtype)

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("_mmaps", None)
        return state

    @property
    def paths(self) -> list[str]:
        """Ordered file paths known to this store."""
        return list(self.entries.keys())

    def _get_mmap(self, path: str, signal: str) -> np.ndarray:
        if not hasattr(self, "_mmaps"):
            self._mmaps: dict[tuple[str, str], np.ndarray] = {}
        key = (path, signal)
        if key not in self._mmaps:
            info = self.entries[path][signal]
            self._mmaps[key] = np.memmap(
                path,
                dtype=info.dtype_str,
                mode="r",
                offset=info.offset,
                shape=info.shape,
            )
        return self._mmaps[key]

    def read_slice(self, path: str, signal: str, l_slc: int, r_slc: int) -> np.ndarray:
        """Read a window from a signal. Thread-safe."""
        path = str(path)
        if self._cache is not None:
            return self._cache[path][signal][l_slc:r_slc].copy()
        info = self.entries[path][signal]
        if info.is_contiguous:
            return np.array(self._get_mmap(path, signal)[l_slc:r_slc], dtype=self.dtype)
        else:
            with h5py.File(path, "r") as f:
                return f[signal][l_slc:r_slc].astype(self.dtype)

    def read_signals(self, path: str, signals: list[str], l_slc: int, r_slc: int) -> np.ndarray:
        """Read and stack multiple signals into shape (window_len, n_signals)."""
        arrays = [self.read_slice(path, s, l_slc, r_slc) for s in signals]
        return np.stack(arrays, axis=-1)

    def get_seq_len(self, path: str, signal: str | None = None) -> int:
        """Get sequence length for a file. Uses first signal if none specified."""
        if signal is None:
            signal = self.signal_names[0]
        return self.entries[str(path)][signal].shape[0]


_HDF_EXTENSIONS = {".hdf5", ".h5"}


def get_hdf_files(path: Path | str) -> list[str]:
    """Get sorted HDF5 file paths from a directory (recursive).

    Returns absolute string paths for all ``*.hdf5`` and ``*.h5`` files.
    """
    return sorted(str(p) for p in Path(path).rglob("*") if p.suffix in _HDF_EXTENSIONS)


def discover_split_files(
    dataset: Path | str,
) -> tuple[list[str], list[str], list[str]]:
    """Discover HDF5 files using the ``train/valid/test`` directory convention.

    Parameters
    ----------
    dataset : Path to dataset root containing ``train/``, ``valid/``, and
        ``test/`` subdirectories.

    Returns
    -------
    train_files, valid_files, test_files : tuple of three sorted file lists.
    """
    root = Path(dataset)
    return (
        get_hdf_files(root / "train"),
        get_hdf_files(root / "valid"),
        get_hdf_files(root / "test"),
    )


def read_hdf5_attr(path: str, key: str, dtype: type = np.float32) -> np.floating:
    """Read a single root-level attribute from an HDF5 file.

    Useful for retrieving per-file metadata such as sampling rate::

        fs = read_hdf5_attr("data.hdf5", "sampling_rate")
    """
    with h5py.File(str(path), "r") as f:
        return dtype(f.attrs[key])
