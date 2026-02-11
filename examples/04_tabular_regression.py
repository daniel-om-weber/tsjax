# %% [markdown]
# # Tabular Regression: Scalar → Scalar
# Demonstrates `ScalarAttr` for both inputs and targets, `RegressionLearner`
# with `MLP`, and the all-scalar pipeline path (no windowing needed).

# %%
from pathlib import Path

from tsjax import RegressionLearner, ScalarAttr, create_grain_dls

# %%
_root = Path(__file__).resolve().parent.parent
DATASET = _root / "test_data/MassSpringDamper"

# %% [markdown]
# ## Build pipeline and train
# Mass-spring-damper features (peak_freq, gain_db, phase_margin) → stiffness.
# Each file has only root attributes — no signal datasets.

# %%
pipeline = create_grain_dls(
    inputs={"u": ScalarAttr(["peak_freq", "gain_db", "phase_margin"])},
    targets={"y": ScalarAttr(["stiffness"])},
    dataset=DATASET,
    bs=16,
)

batch = next(iter(pipeline.train))
print(f"u shape: {batch['u'].shape}")  # (16, 3)
print(f"y shape: {batch['y'].shape}")  # (16, 1)

# %%
lrn = RegressionLearner(pipeline, hidden_sizes=[32, 16])
lrn.fit(n_epoch=5, lr=1e-3)
