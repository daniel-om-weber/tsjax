"""Per-sample data transforms applied via grain.map()."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

Transform = Callable[[np.ndarray], np.ndarray]


def _apply_transforms(transforms: dict[str, Transform]):
    """Return a function that applies per-key transforms to a sample dict."""

    def apply(sample: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        return {k: transforms[k](v) if k in transforms else v for k, v in sample.items()}

    return apply


def stft_transform(n_fft: int = 256, hop_length: int = 128) -> Transform:
    """Create a transform: (seq_len, n_ch) -> (n_frames, n_freq * n_ch).

    Requires scipy to be installed.
    """

    def _stft(x: np.ndarray) -> np.ndarray:
        from scipy.signal import stft

        if x.ndim == 1:
            x = x[:, np.newaxis]
        n_ch = x.shape[1]

        frames = []
        for ch in range(n_ch):
            _, _, Zxx = stft(x[:, ch], nperseg=n_fft, noverlap=n_fft - hop_length)
            frames.append(np.abs(Zxx).T)  # (n_freq, n_frames) -> (n_frames, n_freq)
        return np.concatenate(frames, axis=-1).astype(np.float32)

    return _stft
