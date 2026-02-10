"""Resampling functions and SignalStore wrapper for sample rate conversion.

Provides two resampling strategies matching tsfast:
- resample_interp: fast linear interpolation with Butterworth anti-aliasing
- resample_fft: FFT-based resampling via scipy.signal.resample
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
from scipy.signal import butter, sosfiltfilt
from scipy.signal import resample as scipy_resample

from .store import SignalStore

# Signature for pluggable resampling functions: (signal_1d, factor) -> resampled_1d
ResampleFn = Callable[[np.ndarray, float], np.ndarray]


def resample_interp(
    sig: np.ndarray,
    factor: float,
    lowpass_cut: float = 1.0,
    order: int = 2,
) -> np.ndarray:
    """Resample a 1D signal using linear interpolation with anti-aliasing.

    When downsampling (factor < lowpass_cut), a Butterworth lowpass filter
    is applied before interpolation to prevent aliasing.

    Parameters
    ----------
    sig : 1D array of shape (n,).
    factor : target_fs / source_fs.  < 1 downsamples, > 1 upsamples.
    lowpass_cut : threshold below which the anti-aliasing filter activates.
    order : Butterworth filter order (default 2, matching tsfast).

    Returns
    -------
    Resampled 1D array of shape (round(n * factor),).
    """
    n_orig = len(sig)
    n_target = round(n_orig * factor)
    if n_target == n_orig:
        return sig.copy()
    if n_target <= 0:
        raise ValueError(f"Resampling factor {factor} produces zero-length output")

    work = sig
    if factor < lowpass_cut:
        wn = min(factor, 0.99)  # clamp to avoid invalid Wn=1.0
        sos = butter(order, wn, btype="low", output="sos")
        work = sosfiltfilt(sos, work).astype(sig.dtype)

    x_orig = np.linspace(0, 1, n_orig)
    x_target = np.linspace(0, 1, n_target)
    return np.interp(x_target, x_orig, work).astype(sig.dtype)


def resample_fft(sig: np.ndarray, factor: float) -> np.ndarray:
    """Resample a 1D signal using FFT-based method (scipy.signal.resample).

    Parameters
    ----------
    sig : 1D array of shape (n,).
    factor : target_fs / source_fs.

    Returns
    -------
    Resampled 1D array of shape (round(n * factor),).
    """
    n_orig = len(sig)
    n_target = round(n_orig * factor)
    if n_target == n_orig:
        return sig.copy()
    if n_target <= 0:
        raise ValueError(f"Resampling factor {factor} produces zero-length output")
    return scipy_resample(sig, n_target).astype(sig.dtype)


class ResampledStore:
    """SignalStore wrapper that resamples signals transparently.

    Resamples each signal lazily on first access and caches the result.
    Subsequent reads slice from the cached resampled array.  This matches
    tsfast's resample-then-window approach.

    Fully picklable: the cache is transient and rebuilt after unpickling.
    The *resample_fn* must be a module-level function for picklability.

    Parameters
    ----------
    inner : SignalStore
        The underlying store to read from.
    factor : float or callable
        Resampling factor (target_fs / source_fs).  A float applies the
        same factor to every file.  A callable ``(path: str) -> float``
        returns a per-file factor, enabling multi-rate datasets.
    resample_fn : ResampleFn, optional
        Resampling function ``(signal_1d, factor) -> resampled_1d``.
        Defaults to :func:`resample_interp`.
    """

    def __init__(
        self,
        inner: SignalStore,
        factor: float | Callable[[str], float],
        resample_fn: ResampleFn | None = None,
    ):
        self.inner = inner
        self._factor = factor
        self.resample_fn: ResampleFn = resample_fn or resample_interp
        self._cache: dict[tuple[str, str], np.ndarray] = {}

    def _get_factor(self, path: str) -> float:
        if callable(self._factor):
            return self._factor(path)
        return self._factor

    @property
    def paths(self) -> Sequence[str]:
        return self.inner.paths

    def get_seq_len(self, path: str, signal: str | None = None) -> int:
        """Return the resampled sequence length."""
        orig_len = self.inner.get_seq_len(path, signal)
        return round(orig_len * self._get_factor(path))

    def _get_resampled(self, path: str, signal: str) -> np.ndarray:
        """Get or compute the full resampled signal (lazy cache)."""
        key = (path, signal)
        if key not in self._cache:
            orig_len = self.inner.get_seq_len(path, signal)
            raw = self.inner.read_signals(path, [signal], 0, orig_len)  # (n, 1)
            factor = self._get_factor(path)
            self._cache[key] = self.resample_fn(raw[:, 0], factor)
        return self._cache[key]

    def read_signals(
        self, path: str, signals: list[str], l_slc: int, r_slc: int
    ) -> np.ndarray:
        """Read resampled signals, slicing from cached full sequences."""
        arrays = [self._get_resampled(path, s)[l_slc:r_slc] for s in signals]
        return np.stack(arrays, axis=-1)

    def __getstate__(self):
        """Exclude transient cache for pickling."""
        state = self.__dict__.copy()
        state["_cache"] = {}
        return state
