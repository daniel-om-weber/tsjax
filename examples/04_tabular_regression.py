# %% [markdown]
# # Tabular Regression: Scalar → Scalar
# Demonstrates `ScalarAttr` for both inputs and targets, `RegressionLearner`
# with `MLP`, and the all-scalar pipeline path (no windowing needed).

# %%
import tempfile
from pathlib import Path

import h5py
import numpy as np

from tsjax import RegressionLearner, ScalarAttr, create_grain_dls

# %% [markdown]
# ## Create synthetic dataset
# Mass-spring-damper features → stiffness.
# Each file has only root attributes — no signal datasets.

# %%
rng = np.random.default_rng(42)
tmpdir = tempfile.mkdtemp(prefix="tsjax_reg_")

for split, n_files in [("train", 60), ("valid", 20), ("test", 20)]:
    split_dir = Path(tmpdir) / split
    split_dir.mkdir(parents=True)
    for i in range(n_files):
        stiffness = rng.uniform(10.0, 100.0)
        peak_freq = np.sqrt(stiffness) / (2 * np.pi) + rng.normal(0, 0.1)
        gain_db = 20 * np.log10(1 / stiffness) + rng.normal(0, 0.5)
        phase_margin = 90 - stiffness * 0.3 + rng.normal(0, 2)
        with h5py.File(split_dir / f"{i:03d}.h5", "w") as f:
            f.attrs["peak_freq"] = peak_freq
            f.attrs["gain_db"] = gain_db
            f.attrs["phase_margin"] = phase_margin
            f.attrs["stiffness"] = stiffness

print(f"Dataset: {tmpdir}")

# %% [markdown]
# ## Build pipeline and train

# %%
pipeline = create_grain_dls(
    inputs={"u": ScalarAttr(["peak_freq", "gain_db", "phase_margin"])},
    targets={"y": ScalarAttr(["stiffness"])},
    dataset=tmpdir,
    bs=16,
)

batch = pipeline.train[0]
print(f"u shape: {batch['u'].shape}")  # (16, 3)
print(f"y shape: {batch['y'].shape}")  # (16, 1)

# %%
lrn = RegressionLearner(pipeline, hidden_sizes=[32, 16])
lrn.fit(n_epoch=5, lr=1e-3)
