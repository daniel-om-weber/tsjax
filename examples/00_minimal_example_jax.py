# %% [markdown]
# # Minimal tsjax Example
# JAX equivalent of `00_minimal_example.ipynb`.
# Uses the Wiener-Hammerstein dataset (u → y).

# %%
from pathlib import Path

import grain
import numpy as np

from tsjax import (
    DataSource,
    GrainPipeline,
    HDF5Store,
    NormStats,
    RNNLearner,
    SequenceReader,
    compute_norm_stats_from_index,
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
lrn = RNNLearner(pipeline, rnn_type='lstm', hidden_size=64, n_skip=10, metrics=[rmse])
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
    "u": compute_norm_stats_from_index(store_train, ["u"]),
    "y": compute_norm_stats_from_index(store_train, ["y"]),
}

# 5. Wrap in Grain datasets: shuffle + batch
train_ds = grain.MapDataset.source(train_src).shuffle(seed=42).batch(16, drop_remainder=True)
valid_ds = grain.MapDataset.source(valid_src).batch(16, drop_remainder=False)
test_ds = grain.MapDataset.source(test_src).batch(1, drop_remainder=False)

# 6. Assemble the pipeline (plain dataclass)
pipeline2 = GrainPipeline(
    train=train_ds, valid=valid_ds, test=test_ds,
    stats=stats,
    input_keys=("u",), target_keys=("y",),
    train_source=train_src, valid_source=valid_src, test_source=test_src,
)

# %%
lrn2 = RNNLearner(pipeline2, rnn_type="lstm", hidden_size=64, n_skip=10)
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

stats3 = {s: compute_norm_stats_from_index(stores["train"], [s]) for s in signals}

train_ds3 = grain.MapDataset.source(train_src3).shuffle(seed=42).batch(16, drop_remainder=True)
valid_ds3 = grain.MapDataset.source(valid_src3).batch(16, drop_remainder=False)
test_ds3 = grain.MapDataset.source(test_src3).batch(1, drop_remainder=False)

pipeline3 = GrainPipeline(
    train=train_ds3, valid=valid_ds3, test=test_ds3,
    stats=stats3,
    input_keys=("u",), target_keys=("y",),
    train_source=train_src3, valid_source=valid_src3, test_source=test_src3,
)

# %%
lrn3 = RNNLearner(pipeline3, rnn_type="lstm", hidden_size=64, n_skip=10)
lrn3.fit(n_epoch=1, lr=1e-3)

# %% [markdown]
# ## Adding noise augmentation to training
# Transforms are per-sample functions inserted via `.map()` before batching.
# Apply only to the training dataset — valid/test stay clean.

# %%
rng = np.random.default_rng(0)


def add_noise(sample):
    return {k: v + rng.normal(0, 0.01, v.shape).astype(v.dtype) for k, v in sample.items()}


train_ds4 = (
    grain.MapDataset.source(train_src3).map(add_noise).shuffle(seed=42)
    .batch(16, drop_remainder=True)
)

pipeline4 = GrainPipeline(
    train=train_ds4, valid=valid_ds3, test=test_ds3,
    stats=stats3,
    input_keys=("u",), target_keys=("y",),
    train_source=train_src3, valid_source=valid_src3, test_source=test_src3,
)

# %%
lrn4 = RNNLearner(pipeline4, rnn_type="lstm", hidden_size=64, n_skip=10)
lrn4.fit(n_epoch=1, lr=1e-3)
