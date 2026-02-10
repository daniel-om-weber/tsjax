# %% [markdown]
# # Classification: Sequence → Class Label
# Demonstrates `ScalarAttr` targets, `ClassifierLearner`, and
# classification on synthetic damped sinusoids with 3 damping regimes.

# %%
from pathlib import Path

from tsjax import ClassifierLearner, ScalarAttr, create_grain_dls

# %%
_root = Path(__file__).resolve().parent.parent
DATASET = _root / "test_data/DampedSinusoids"

# %% [markdown]
# ## Build pipeline and train
# Three damping regimes (class 0, 1, 2) with per-file `"class"` attribute
# and windowed `"u"` signal (damped sinusoid, 1000 pts per file).

# %%
pipeline = create_grain_dls(
    inputs={"u": ["u"]},
    targets={"y": ScalarAttr(["class"])},
    dataset=DATASET,
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
