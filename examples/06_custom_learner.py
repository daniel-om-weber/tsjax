# %% [markdown]
# # Custom Model and Learner
# Demonstrates `Feature` spec for reducing a signal window to a scalar,
# and manual `Learner` construction with composable model layers.

# %%
from pathlib import Path

import numpy as np
from flax import nnx

from tsjax import (
    RNN,
    Denormalize,
    Feature,
    LastPool,
    Learner,
    Normalize,
    NormalizedModel,
    create_grain_dls,
    normalized_mse,
)

# %%
DATASET = Path(__file__).resolve().parent.parent / "test_data/WienerHammerstein"

# %% [markdown]
# ## Feature target
# Reduce a `(win_sz, n_ch)` window to a scalar `(1,)` via a custom function.


# %%
def rms_feature(y: np.ndarray) -> np.ndarray:
    """RMS of the signal window: (win_sz, 1) -> (1,)."""
    return np.sqrt(np.mean(y**2, axis=0))


pipeline = create_grain_dls(
    inputs={"u": ["u"]},
    targets={"y": Feature(["y"], fn=rms_feature)},
    dataset=DATASET,
    win_sz=500, stp_sz=10, bs=16,
    preload=True,
)

batch = next(iter(pipeline.train))
print(f"u: {batch['u'].shape}")  # (16, 500, 1) — windowed signal
print(f"y: {batch['y'].shape}")  # (16, 1)       — scalar per window

# %% [markdown]
# ## Manual model construction
# `RNN + LastPool` reduces sequence → scalar. `NormalizedModel` wraps
# the model with input normalization and output denormalization.

# %%
u_stats = pipeline.stats["u"]
y_stats = pipeline.stats["y"]

rnn = RNN(input_size=1, output_size=1, hidden_size=64, rngs=nnx.Rngs(0))
model = NormalizedModel(
    nnx.Sequential(rnn, LastPool()),
    norm_in=Normalize(1, mean=u_stats.mean, std=u_stats.std),
    norm_out=Denormalize(1, mean=y_stats.mean, std=y_stats.std),
)

# %%
lrn = Learner(model, pipeline, loss_func=normalized_mse)
lrn.fit(n_epoch=3, lr=1e-3)
