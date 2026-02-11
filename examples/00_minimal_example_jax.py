# %% [markdown]
# # Minimal tsjax Example
# JAX equivalent of `00_minimal_example.ipynb`.
# Uses the Wiener-Hammerstein dataset (u → y).

# %%
from pathlib import Path

import numpy as np

from tsjax import (
    DataSource,
    GrainPipeline,
    HDF5Store,
    NormStats,
    RNNLearner,
    SequenceReader,
    compute_stats,
    create_simulation_dls,
    rmse,
)

# %%
_root = Path(__file__).resolve().parent.parent
DATASET = _root / "test_data/WienerHammerstein"

pipeline = create_simulation_dls(
    u=['u'], y=['y'],
    dataset=DATASET,
    bs=16, win_sz=500, stp_sz=10,
    preload=True
)

# %%
lrn = RNNLearner(pipeline, hidden_size=64, n_skip=10, metrics=[rmse])
lrn.fit_flat_cos(n_epoch=1, lr=1e-3)

# %% [markdown]
# ## Alternative: fully explicit pipeline
# `create_simulation_dls` assembles these components under the hood.
# Use this when you need custom stores, readers, or non-standard splits.

# %%
# 1. Discover files per split
train_files = sorted(str(p) for p in (DATASET / "train").rglob("*.hdf5"))
valid_files = sorted(str(p) for p in (DATASET / "valid").rglob("*.hdf5"))
test_files = sorted(str(p) for p in (DATASET / "test").rglob("*.hdf5"))

# 2. Create stores (format-level access to signal data)
store_train = HDF5Store(train_files, ["u", "y"], preload=True)
store_valid = HDF5Store(valid_files, ["u", "y"], preload=True)
store_test = HDF5Store(test_files, ["u", "y"], preload=True)

# 3. Build DataSources with explicit readers
train_src = DataSource(store_train, {
    "u": SequenceReader(store_train, ["u"]),
    "y": SequenceReader(store_train, ["y"]),
}, win_sz=500, stp_sz=10)

valid_src = DataSource(store_valid, {
    "u": SequenceReader(store_valid, ["u"]),
    "y": SequenceReader(store_valid, ["y"]),
}, win_sz=500, stp_sz=10)

test_src = DataSource(store_test, {
    "u": SequenceReader(store_test, ["u"]),
    "y": SequenceReader(store_test, ["y"]),
})  # full sequence — no win_sz

# 4. Compute normalization stats from training data
stats: dict[str, NormStats] = {
    "u": compute_stats(train_src, "u"),
    "y": compute_stats(train_src, "y"),
}

# 5. Assemble the pipeline via from_sources
pipeline2 = GrainPipeline.from_sources(
    train_src, valid_src, test_src,
    input_keys=("u",), target_keys=("y",),
    bs=16, seed=42, stats=stats,
)

# %%
lrn2 = RNNLearner(pipeline2, hidden_size=64, n_skip=10)
lrn2.fit(n_epoch=1, lr=1e-3)

# %% [markdown]
# ## Compact variant: same building blocks, less repetition

# %%
signals = ["u", "y"]
stores = {
    s: HDF5Store(sorted(str(p) for p in (DATASET / s).rglob("*.hdf5")), signals, preload=True)
    for s in ("train", "valid", "test")
}


def make_source(store, **kw):
    return DataSource(store, {
        "u": SequenceReader(store, [signals[0]]),
        "y": SequenceReader(store, [signals[1]])}, **kw)


train_src3 = make_source(stores["train"], win_sz=500, stp_sz=10)
valid_src3 = make_source(stores["valid"], win_sz=500, stp_sz=10)
test_src3 = make_source(stores["test"])

stats3 = {s: compute_stats(train_src3, s) for s in signals}

pipeline3 = GrainPipeline.from_sources(
    train_src3, valid_src3, test_src3,
    input_keys=("u",), target_keys=("y",),
    bs=16, seed=42, stats=stats3,
)

# %%
lrn3 = RNNLearner(pipeline3, hidden_size=64, n_skip=10)
lrn3.fit(n_epoch=1, lr=1e-3)

# %% [markdown]
# ## Adding noise augmentation to training
# Augmentations are per-key random transforms applied only to training data.

# %%
from tsjax import noise_injection  # noqa: E402

pipeline4 = GrainPipeline.from_sources(
    train_src3, valid_src3, test_src3,
    input_keys=("u",), target_keys=("y",),
    bs=16, seed=42, stats=stats3,
    augmentations={"u": noise_injection(0.01)},
)

# %%
lrn4 = RNNLearner(pipeline4, hidden_size=64, n_skip=10)
lrn4.fit(n_epoch=1, lr=1e-3)
