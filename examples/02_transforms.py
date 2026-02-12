# %% [markdown]
# # Transforms and Augmentations
# Per-sample transforms (deterministic, all splits) and augmentations
# (random, training-only). Uses the Wiener-Hammerstein dataset (u → y).

# %%
from pathlib import Path

import numpy as np

from tsjax import RNNLearner, create_grain_dls, rmse, stft_transform

# %%
DATASET = Path(__file__).resolve().parent.parent / "test_data/WienerHammerstein"

# %% [markdown]
# ## Custom transform
# A transform is any function `(sample_dict) -> sample_dict` applied per-sample
# before batching. Stats are computed on transformed data.


# %%
def cumsum_transform(x: np.ndarray) -> np.ndarray:
    """Cumulative sum: (seq_len, n_ch) -> (seq_len, n_ch)."""
    return np.cumsum(x, axis=0).astype(np.float32)


pipeline = create_grain_dls(
    inputs={"u": ["u"]},
    targets={"y": ["y"]},
    dataset=DATASET,
    bs=16,
    win_sz=500,
    stp_sz=10,
    preload=True,
    transform=lambda item: {**item, "u": cumsum_transform(item["u"])},
)

print(f"u stats (cumsum): mean={pipeline.stats()['u'].mean}, std={pipeline.stats()['u'].std}")

# %% [markdown]
# ## Built-in STFT transform
# Shape-changing transform: `(seq_len, n_ch)` → `(n_frames, n_freq * n_ch)`.

# %%
_stft = stft_transform(n_fft=64, hop_length=32)

pipeline_stft = create_grain_dls(
    inputs={"u": ["u"]},
    targets={"y": ["y"]},
    dataset=DATASET,
    bs=16,
    win_sz=500,
    stp_sz=10,
    preload=True,
    transform=lambda item: {**item, "u": _stft(item["u"])},
)

batch = next(iter(pipeline_stft.train))
print(f"u shape after STFT: {batch['u'].shape}")  # (16, n_frames, 33)

# %% [markdown]
# ## Augmentations
# Random transforms applied only to training data. They don't affect
# normalization stats. Compose multiple augmentations with `chain_augmentations`.

# %%
from tsjax import bias_injection, chain_augmentations, noise_injection, varying_noise  # noqa: E402

_aug_u = chain_augmentations(
    noise_injection(0.01),
    bias_injection(0.005),
    varying_noise(0.02),
)

pipeline_aug = create_grain_dls(
    inputs={"u": ["u"]},
    targets={"y": ["y"]},
    dataset=DATASET,
    bs=16,
    win_sz=500,
    stp_sz=10,
    preload=True,
    augmentation=lambda item, rng: {**item, "u": _aug_u(item["u"], rng)},
)

# Stats are identical — augmentations don't affect them
print(f"u mean (no aug):   {pipeline.stats()['u'].mean}")
print(f"u mean (with aug): {pipeline_aug.stats()['u'].mean}")

# %%
lrn = RNNLearner(pipeline_aug, hidden_size=64, n_skip=10, metrics=[rmse])
lrn.fit_flat_cos(n_epoch=1, lr=1e-3)
