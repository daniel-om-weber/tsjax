# %% [markdown]
# # Data Pipeline
# How tsjax loads data: from the high-level `create_grain_dls` factory
# down to explicit `HDF5Store` → `WindowedSource` → `GrainPipeline` construction.

# %%
from pathlib import Path

from tsjax import (
    GrainPipeline,
    HDF5Store,
    RNNLearner,
    WindowedSource,
    create_grain_dls,
    rmse,
)

# %%
DATASET = Path(__file__).resolve().parent.parent / "test_data/WienerHammerstein"

# %% [markdown]
# ## create_grain_dls — the flexible factory

# %%
pipeline = create_grain_dls(
    inputs={"u": ["u"]},
    targets={"y": ["y"]},
    dataset=DATASET,
    bs=16, win_sz=500, stp_sz=10,
    valid_stp_sz=500,  # non-overlapping validation windows
    preload=True,
)

batch = next(iter(pipeline.train))
print(f"u: {batch['u'].shape}, y: {batch['y'].shape}")
print(f"u stats: mean={pipeline.stats['u'].mean}, std={pipeline.stats['u'].std}")

# %% [markdown]
# ## Multi-signal inputs
# Pass multiple signal names to read them as channels in a single key.

# %%
pipeline_multi = create_grain_dls(
    inputs={"u": ["u", "y"]},  # 2-channel input
    targets={"y": ["y"]},
    dataset=DATASET,
    bs=16, win_sz=500, stp_sz=10,
    preload=True,
)

batch = next(iter(pipeline_multi.train))
print(f"u: {batch['u'].shape}")  # (16, 500, 2)

# %% [markdown]
# ## Explicit pipeline construction
# `create_grain_dls` assembles these components under the hood.
# Use this when you need custom stores, readers, or non-standard splits.

# %%
train_files = sorted(str(p) for p in (DATASET / "train").rglob("*.hdf5"))
valid_files = sorted(str(p) for p in (DATASET / "valid").rglob("*.hdf5"))
test_files = sorted(str(p) for p in (DATASET / "test").rglob("*.hdf5"))

store_train = HDF5Store(train_files, ["u", "y"], preload=True)
store_valid = HDF5Store(valid_files, ["u", "y"], preload=True)
store_test = HDF5Store(test_files, ["u", "y"], preload=True)

train_src = WindowedSource(store_train, {"u": ["u"], "y": ["y"]}, win_sz=500, stp_sz=10)
valid_src = WindowedSource(store_valid, {"u": ["u"], "y": ["y"]}, win_sz=500, stp_sz=10)
test_src = WindowedSource(store_test, {"u": ["u"], "y": ["y"]})  # full files

pipeline_explicit = GrainPipeline.from_sources(
    train_src, valid_src, test_src,
    input_keys=("u",), target_keys=("y",),
    bs=16, seed=42,
)

# %% [markdown]
# ## Visualize a batch

# %%
lrn = RNNLearner(pipeline, hidden_size=64, n_skip=10, metrics=[rmse])
lrn.show_batch(n=4)
