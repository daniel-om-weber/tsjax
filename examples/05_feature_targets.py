# %% [markdown]
# # Feature Targets: Windowed Signals → Derived Scalar
# Demonstrates `Feature` spec for reducing a signal window to a scalar,
# and manual `Learner` construction for custom model/loss combos.
# Uses existing Wiener-Hammerstein test data.

# %%
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from flax import nnx

from tsjax import Feature, Learner, RNNEncoder, create_grain_dls, normalized_mse

# %%
_root = Path(__file__).resolve().parent.parent
DATASET = _root / "test_data/WienerHammerstein"


# %% [markdown]
# ## Define a feature function
# Reduces a `(win_sz, n_ch)` window to a scalar `(1,)`.

# %%
def rms_feature(y: np.ndarray) -> np.ndarray:
    """RMS of the signal window: (win_sz, 1) -> (1,)."""
    return np.sqrt(np.mean(y**2, axis=0))


# %% [markdown]
# ## Build pipeline with Feature target

# %%
pipeline = create_grain_dls(
    inputs={"u": ["u"]},
    targets={"y": Feature(["y"], fn=rms_feature)},
    dataset=DATASET,
    win_sz=500,
    stp_sz=10,
    bs=16,
    preload=True,
)

batch = pipeline.train[0]
print(f"u shape: {batch['u'].shape}")  # (16, 500, 1) — windowed signal
print(f"y shape: {batch['y'].shape}")  # (16, 1)       — scalar per window

# %% [markdown]
# ## Manual Learner construction
# `RNNEncoder` with y stats from the pipeline — the model denormalizes
# its output to physical units, so we can use `normalized_mse` directly.

# %%
u_stats = pipeline.stats["u"]
y_stats = pipeline.stats["y"]
model = RNNEncoder(
    input_size=1,
    output_size=1,
    hidden_size=64,
    u_mean=jnp.asarray(u_stats.mean),
    u_std=jnp.asarray(u_stats.std),
    y_mean=jnp.asarray(y_stats.mean),
    y_std=jnp.asarray(y_stats.std),
    rngs=nnx.Rngs(0),
)

lrn = Learner(model, pipeline, loss_func=normalized_mse)
lrn.fit(n_epoch=1, lr=1e-3)
