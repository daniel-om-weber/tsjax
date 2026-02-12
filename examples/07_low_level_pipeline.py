# %% [markdown]
# # Low-Level Pipeline
# Maximum flexibility: build the entire data pipeline from individual
# components, bypassing all factory functions and `GrainPipeline`.
#
# This shows every layer of the stack:
# `HDF5Store` → `WindowedSource` → `grain.MapDataset` chain

# %%
from pathlib import Path

import grain
import numpy as np

from tsjax import (
    HDF5Store,
    WindowedSource,
    discover_split_files,
    noise_injection,
)

# %%
DATASET = Path(__file__).resolve().parent.parent / "test_data/WienerHammerstein"

# %% [markdown]
# ## 1. Storage layer — HDF5Store
# One store per split.  `preload=True` reads everything into RAM;
# `preload=False` uses memory-mapped reads (better for large datasets).

# %%
train_files, valid_files, test_files = discover_split_files(DATASET)

signals = ["u", "y"]
store_train = HDF5Store(train_files, signals, preload=True)
store_valid = HDF5Store(valid_files, signals, preload=True)
store_test  = HDF5Store(test_files,  signals, preload=True)

# %% [markdown]
# ## 2. Sources — Grain-compatible `__getitem__`
# `WindowedSource` yields sliding windows (when `win_sz` is set)
# or full files (when `win_sz=None`).

# %%
WIN_SZ, STP_SZ, BS = 500, 10, 16

train_src = WindowedSource(store_train, {"u": ["u"], "y": ["y"]}, win_sz=WIN_SZ, stp_sz=STP_SZ)
valid_src = WindowedSource(store_valid, {"u": ["u"], "y": ["y"]}, win_sz=WIN_SZ, stp_sz=WIN_SZ)
test_src  = WindowedSource(store_test, {"u": ["u"], "y": ["y"]})  # full files

print(f"Train windows: {len(train_src)}, Valid: {len(valid_src)}, Test files: {len(test_src)}")

# Inspect a single sample
sample = train_src[0]
print(f"Sample keys: {list(sample)}")
print(f"u: {sample['u'].shape}, y: {sample['y'].shape}")

# %% [markdown]
# ## 3. grain.MapDataset — build the iteration chain
# This is what `GrainPipeline.from_sources` does internally.
# Doing it yourself gives full control over shuffle, repeat, transforms,
# augmentations, batching, and prefetching.

# %%
# -- Augmentation: additive noise on u during training only --
augment_u = noise_injection(std=0.05)

def train_augment(sample, rng):
    return {**sample, "u": augment_u(sample["u"], rng)}

# -- Train: shuffled, infinite, augmented --
train_ds = (
    grain.MapDataset.source(train_src)
    .seed(42)
    .shuffle()
    .repeat(None)            # infinite cycling
    .random_map(train_augment)
    .batch(BS, drop_remainder=True)
    .to_iter_dataset()
)

# -- Valid: sequential, finite --
valid_ds = (
    grain.MapDataset.source(valid_src)
    .batch(BS, drop_remainder=False)
    .to_iter_dataset()
)

# -- Test: one full sequence at a time --
test_ds = (
    grain.MapDataset.source(test_src)
    .batch(1, drop_remainder=False)
    .to_iter_dataset()
)

# %% [markdown]
# ## 4. Wrap in GrainPipeline
# The hand-built datasets slot straight into `GrainPipeline`.
# This gives you `.stats`, `.n_train_batches`, etc.

# %%
from tsjax import GrainPipeline

pipeline = GrainPipeline(
    train=train_ds,
    valid=valid_ds,
    test=test_ds,
    input_keys=("u",),
    target_keys=("y",),
    train_source=train_src,
    valid_source=valid_src,
    test_source=test_src,
    bs=BS,
)

batch = next(iter(pipeline.train))
print(f"Train batch — u: {batch['u'].shape}, y: {batch['y'].shape}")

for key, s in pipeline.stats.items():
    print(f"{key}: mean={s.mean}, std={s.std}")

# %% [markdown]
# ## 5. Custom transforms via grain `.map()`
# Any per-sample computation can be added as a grain transform.
# Here's a transform that adds the RMS energy of u as an extra feature.

# %%
def add_rms_feature(sample):
    """Compute RMS energy of u and add as a new key."""
    u = sample["u"]
    rms = np.sqrt(np.mean(u ** 2, axis=0, keepdims=True))
    return {**sample, "u_rms": rms.astype(np.float32)}

custom_ds = (
    grain.MapDataset.source(train_src)
    .map(add_rms_feature)
    .batch(BS, drop_remainder=True)
    .to_iter_dataset()
)

batch = next(iter(custom_ds))
print(f"Custom transform — u: {batch['u'].shape}, u_rms: {batch['u_rms'].shape}")
