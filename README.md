# tsjax

JAX-based training library for time series system identification.

## Overview

tsjax reimplements the [TSFast](https://github.com/daniel-at/tsfast) training pipeline in JAX.
It provides the same Learner factory pattern using Flax NNX models, Grain data loading, and Optax optimization.
Models accept raw physical values and handle normalization internally (raw-in/raw-out).

## Installation

```bash
pip install tsjax
```

For development:

```bash
git clone https://github.com/daniel-at/tsjax.git
cd tsjax
uv pip install -e ".[test]"
```

## Quick Start

```python
from tsjax import create_grain_dls, GRULearner, rmse

pipeline = create_grain_dls(
    u=['u'], y=['y'],
    dataset='path/to/dataset',
    bs=16, win_sz=500, stp_sz=10,
)

lrn = GRULearner(pipeline, hidden_size=64, metrics=[rmse])
lrn.fit_flat_cos(n_epoch=10, lr=1e-3)
```

## Dataset Format

Datasets are organized as directories with `train/`, `valid/`, and `test/` subdirectories, each containing HDF5 files with signal arrays:

```
my_dataset/
├── train/
│   └── data.hdf5    # contains arrays: u, y (or custom signal names)
├── valid/
│   └── data.hdf5
└── test/
    └── data.hdf5
```

HDF5 datasets should use contiguous layout (h5py default) for optimal mmap performance.

## Features

- **RNN/GRU/LSTM models** with internal normalization — raw physical values in, raw physical values out
- **Grain data pipeline** with mmap HDF5 reading and computed windowing
- **Learner** with `fit()` and `fit_flat_cos()` training schedules
- **Loss functions:** `normalized_mse`, `normalized_mae`, `rmse` (per-channel normalized space)
- **Factory functions:** `RNNLearner()`, `GRULearner()` for quick setup

## Architecture

See [DESIGN.md](DESIGN.md) for architecture decisions and TSFast comparison.

## License

Apache 2.0
