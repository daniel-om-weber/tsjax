# %% [markdown]
# # Classification: Sequence → Class Label
# Demonstrates `ScalarAttr` targets and `ClassifierLearner` on synthetic
# damped sinusoids with 3 damping regimes.

# %%
from pathlib import Path

from tsjax import ClassifierLearner, ScalarAttr, create_grain_dls

# %%
DATASET = Path(__file__).resolve().parent.parent / "test_data/DampedSinusoids"

# %%
pipeline = create_grain_dls(
    inputs={"u": ["u"]},
    targets={"y": ScalarAttr(["class"])},
    dataset=DATASET,
    win_sz=500, stp_sz=250, bs=8,
    preload=True,
)

# %%
lrn = ClassifierLearner(pipeline, n_classes=3, hidden_size=64)
lrn.show_batch(n=4)

# %%
lrn.fit(n_epoch=3, lr=1e-3)

# %%
lrn.show_results(n=4)
