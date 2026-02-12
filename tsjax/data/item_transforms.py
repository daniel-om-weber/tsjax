"""Per-sample data transforms and training-only augmentations."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

# ---------------------------------------------------------------------------
# Deterministic transforms (all splits, affects stats)
# ---------------------------------------------------------------------------


def stft_transform(n_fft: int = 256, hop_length: int = 128) -> Callable[[np.ndarray], np.ndarray]:
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


def seq_slice(
    l_slc: int | None = None, r_slc: int | None = None
) -> Callable[[np.ndarray], np.ndarray]:
    """Create a slicing transform: ``x[l_slc:r_slc]``.

    Useful for temporal shifting between input and output signals.
    """

    def _slice(x: np.ndarray) -> np.ndarray:
        return x[l_slc:r_slc]

    return _slice


# ---------------------------------------------------------------------------
# Random augmentations (train only, no stats impact)
# ---------------------------------------------------------------------------


def noise_injection(
    std: float | np.ndarray = 0.1,
    mean: float | np.ndarray = 0.0,
    p: float = 1.0,
) -> Callable[[np.ndarray, np.random.Generator], np.ndarray]:
    """Per-channel additive Gaussian noise.

    Parameters
    ----------
    std : Noise standard deviation. Scalar or ``(n_channels,)`` array.
    mean : Noise mean. Scalar or ``(n_channels,)`` array.
    p : Probability of applying the augmentation per sample.
    """
    std = np.asarray(std, dtype=np.float32)
    mean = np.asarray(mean, dtype=np.float32)

    def _augment(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        if p < 1.0 and rng.random() > p:
            return x
        return x + rng.normal(mean, std, x.shape).astype(x.dtype)

    return _augment


def varying_noise(
    std_std: float = 0.1, p: float = 1.0
) -> Callable[[np.ndarray, np.random.Generator], np.ndarray]:
    """Noise with randomly-sampled standard deviation.

    Samples ``std = abs(normal(0, std_std))`` per application, then adds
    ``normal(0, std, shape)``.

    Parameters
    ----------
    std_std : Standard deviation of the std sampling distribution.
    p : Probability of applying the augmentation per sample.
    """

    def _augment(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        if p < 1.0 and rng.random() > p:
            return x
        std = np.abs(rng.normal(0, std_std))
        return x + rng.normal(0, std, x.shape).astype(x.dtype)

    return _augment


def grouped_noise(
    std_std: np.ndarray,
    std_idx: np.ndarray,
    p: float = 1.0,
) -> Callable[[np.ndarray, np.random.Generator], np.ndarray]:
    """Per-group noise with random std per group.

    Each channel belongs to a group (via *std_idx*). Per-group noise
    standard deviations are sampled from ``abs(normal(0, std_std[group]))``.

    Parameters
    ----------
    std_std : ``(n_groups,)`` — std of the std distribution per group.
    std_idx : ``(n_channels,)`` — maps each channel to its group index.
    p : Probability of applying the augmentation per sample.
    """
    std_std = np.asarray(std_std, dtype=np.float32)
    std_idx = np.asarray(std_idx, dtype=np.int64)

    def _augment(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        if p < 1.0 and rng.random() > p:
            return x
        group_stds = np.abs(rng.normal(0, std_std))  # (n_groups,)
        per_ch_std = group_stds[std_idx]  # (n_channels,)
        return x + rng.normal(0, per_ch_std, x.shape).astype(x.dtype)

    return _augment


def bias_injection(
    std: float | np.ndarray = 0.1,
    mean: float | np.ndarray = 0.0,
    p: float = 1.0,
) -> Callable[[np.ndarray, np.random.Generator], np.ndarray]:
    """Per-sample constant offset broadcast across time.

    Generates one random offset per channel with shape ``(1, n_ch)``
    and broadcasts it across the time dimension.

    Parameters
    ----------
    std : Offset standard deviation. Scalar or ``(n_channels,)`` array.
    mean : Offset mean. Scalar or ``(n_channels,)`` array.
    p : Probability of applying the augmentation per sample.
    """
    std = np.asarray(std, dtype=np.float32)
    mean = np.asarray(mean, dtype=np.float32)

    def _augment(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        if p < 1.0 and rng.random() > p:
            return x
        offset = rng.normal(mean, std, (1, x.shape[-1])).astype(x.dtype)
        return x + offset

    return _augment


def chain_augmentations(
    *augs: Callable[[np.ndarray, np.random.Generator], np.ndarray],
) -> Callable[[np.ndarray, np.random.Generator], np.ndarray]:
    """Compose multiple augmentations left-to-right, sharing the same *rng*."""

    def _chained(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        for aug in augs:
            x = aug(x, rng)
        return x

    return _chained
