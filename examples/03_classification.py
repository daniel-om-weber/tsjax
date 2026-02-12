# %% [markdown]
# # Classification: Sequence → Class Label
# Demonstrates `scalar_attrs` for per-file targets and `ClassifierLearner`
# on synthetic damped sinusoids with 3 damping regimes.

# %%
from pathlib import Path

from tsjax import (
    ClassifierLearner,
    GrainPipeline,
    HDF5Store,
    WindowedSource,
    discover_split_files,
    scalar_attrs,
)

# %%
DATASET = Path(__file__).resolve().parent.parent / "test_data/DampedSinusoids"

# %% [markdown]
# ## Build pipeline and train
# Sequential input "u" with per-file scalar target "class".

# %%
train_files, valid_files, test_files = discover_split_files(DATASET)

signals = ["u"]

store_train = HDF5Store(train_files, signals, preload=True)
store_valid = HDF5Store(valid_files, signals, preload=True)
store_test = HDF5Store(test_files, signals, preload=True)

train_src = WindowedSource(
    store_train,
    {"u": ["u"], "y": scalar_attrs(train_files, ["class"])},
    win_sz=500,
    stp_sz=250,
)
valid_src = WindowedSource(
    store_valid,
    {"u": ["u"], "y": scalar_attrs(valid_files, ["class"])},
    win_sz=500,
    stp_sz=500,
)
test_src = WindowedSource(
    store_test,
    {"u": ["u"], "y": scalar_attrs(test_files, ["class"])},
)

pipeline = GrainPipeline.from_sources(
    train_src,
    valid_src,
    test_src,
    input_keys=("u",),
    target_keys=("y",),
    bs=8,
    seed=42,
)

# %%
lrn = ClassifierLearner(pipeline, n_classes=3, hidden_size=64)
lrn.show_batch(n=4)

# %%
lrn.fit(n_epoch=3, lr=1e-3)

# %%
lrn.show_results(n=4)
