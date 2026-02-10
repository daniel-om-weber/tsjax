# %% [markdown]
# # Using Transforms
# Demonstrates per-key data transforms applied before batching.
# Uses the Wiener-Hammerstein dataset (u → y) with a cumulative-sum
# transform on the input signal — turning the raw excitation into
# a running integral the model sees instead of the raw values.

# %%
from pathlib import Path

import numpy as np

from tsjax import RNNLearner, create_grain_dls, rmse

# %%
_root = Path(__file__).resolve().parent.parent
DATASET = _root / "test_data/WienerHammerstein"


# %% [markdown]
# ## Define a transform
# A transform is any `(np.ndarray) -> np.ndarray` callable.
# It receives a single sample array and returns the transformed array.
# Shape-preserving transforms work with the standard simulation pipeline;
# shape-changing transforms (e.g., STFT) are also supported.

# %%
def cumsum_transform(x: np.ndarray) -> np.ndarray:
    """Cumulative sum: (seq_len, n_ch) -> (seq_len, n_ch)."""
    return np.cumsum(x, axis=0).astype(np.float32)


def clip_transform(x: np.ndarray) -> np.ndarray:
    """Clip values to [-1, 1]."""
    return np.clip(x, -1.0, 1.0).astype(np.float32)


# %% [markdown]
# ## Build pipeline with transform
# Pass `transforms={"u": cumsum_transform}` to apply the transform
# to the `"u"` key on every sample before batching.  Norm stats are
# automatically computed on the *transformed* data.

# %%
pipeline = create_grain_dls(
    inputs={"u": ["u"]},
    targets={"y": ["y"]},
    dataset=DATASET,
    bs=16,
    win_sz=500,
    stp_sz=10,
    preload=True,
    # Chain multiple transforms on the same key with a lambda:
    transforms={"u": lambda x: clip_transform(cumsum_transform(x))},
)

batch = pipeline.train[0]
print(f"u shape: {batch['u'].shape}")  # (16, 500, 1) — same shape, different values
print(f"y shape: {batch['y'].shape}")  # (16, 500, 1) — unchanged

# Stats reflect the transformed (cumsum then clipped) data
print(f"u mean: {pipeline.stats['u'].mean}")
print(f"u std:  {pipeline.stats['u'].std}")

# %% [markdown]
# ## Train with transformed inputs

# %%
lrn = RNNLearner(pipeline, hidden_size=64, n_skip=10, metrics=[rmse])
lrn.fit_flat_cos(n_epoch=1, lr=1e-3)
