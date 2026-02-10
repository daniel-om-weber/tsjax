# %% [markdown]
# # Classification: Sequence → Class Label
# Demonstrates `ScalarAttr` targets, `ClassifierLearner`, and
# classification on synthetic damped sinusoids with 3 damping regimes.

# %%
import tempfile
from pathlib import Path

import h5py
import numpy as np

from tsjax import ClassifierLearner, ScalarAttr, create_grain_dls

# %% [markdown]
# ## Create synthetic dataset
# Three damping regimes: underdamped (0), critically damped (1), overdamped (2).
# Each file: `"u"` signal (1000 pts) + `"class"` root attribute.

# %%
rng = np.random.default_rng(42)
tmpdir = tempfile.mkdtemp(prefix="tsjax_cls_")
DAMPING = {0: 0.02, 1: 0.10, 2: 0.30}

for split, n_files in [("train", 30), ("valid", 10), ("test", 10)]:
    split_dir = Path(tmpdir) / split
    split_dir.mkdir(parents=True)
    for i in range(n_files):
        cls = int(rng.integers(0, 3))
        t = np.linspace(0, 10, 1000, dtype=np.float32)
        u = np.exp(-DAMPING[cls] * t) * np.sin(2 * np.pi * t + rng.uniform(0, 2 * np.pi))
        with h5py.File(split_dir / f"{i:03d}.h5", "w") as f:
            f.create_dataset("u", data=u.astype(np.float32))
            f.attrs["class"] = cls

print(f"Dataset: {tmpdir}")

# %% [markdown]
# ## Build pipeline and train

# %%
pipeline = create_grain_dls(
    inputs={"u": ["u"]},
    targets={"y": ScalarAttr(["class"])},
    dataset=tmpdir,
    win_sz=500,
    stp_sz=250,
    bs=8,
    preload=True,
)

batch = pipeline.train[0]
print(f"u shape: {batch['u'].shape}")  # (8, 500, 1)
print(f"y shape: {batch['y'].shape}")  # (8, 1)

# %%
lrn = ClassifierLearner(pipeline, n_classes=3, hidden_size=64)
lrn.fit(n_epoch=3, lr=1e-3)
