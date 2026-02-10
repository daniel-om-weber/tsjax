# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

tsjax is a JAX-based training library for time series system identification, a standalone sibling of [TSFast](https://github.com/daniel-at/tsfast) (PyTorch/fastai). It reimplements the training pipeline using Flax NNX models, Grain data loading, and Optax optimization while maintaining numerical parity with TSFast. The two libraries cannot coexist in the same process (grain/sklearn mutex conflict).

## Commands

```bash
# Install in development mode
uv pip install -e ".[test]"

# Lint and format
ruff check tsjax/ tests/
ruff format --check tsjax/ tests/    # check only
ruff format tsjax/ tests/            # auto-fix

# Run all tests
pytest tests/ -v

# Run a single test file or test
pytest tests/test_models.py -v
pytest tests/test_training.py::test_training_reduces_loss -v

# Run minimal example
python examples/00_minimal_example_jax.py
```

## Architecture

**Data flow** (spans multiple modules):
```
HDF5 files → HDF5MmapIndex (byte offsets via h5py at init, np.memmap at runtime)
  → WindowedSource (on-the-fly windowing via bisect, no precomputed tuples)
  → Grain MapDataset (shuffle + batch → {'u': array, 'y': array} dicts)
  → Learner._fit() unpacks batch, feeds arrays to model
  → RNN model (raw-in → normalize → multi-layer RNN → linear → denormalize → raw-out)
  → Loss computed in normalized space (per-channel balanced gradients)
  → Optax optimizer updates params
```

**Key design pattern — model-internal normalization:** Unlike TSFast (which normalizes in the data pipeline), tsjax models are self-contained raw-in/raw-out. Norm stats are stored as `Buffer` variables (custom NNX Variable type in `_core.py`, excluded from gradients). Loss functions receive raw predictions and normalize internally. This means no external denormalization is needed at inference time.

**Factory pattern:** `RNNLearner(pipeline, ...)` / `GRULearner(pipeline, ...)` in `training/factory.py` create a model from pipeline norm stats, then wrap it in a `Learner`. This mirrors TSFast's API. `GRULearner` is a `functools.partial` alias.

**Package structure:**
```
tsjax/
    _core.py             # Buffer(nnx.Variable) — shared framework primitive
    data/
        index.py         # SignalIndex Protocol — format-agnostic interface
        hdf5_index.py    # HDF5MmapIndex — HDF5/mmap implementation of SignalIndex
        sources.py       # WindowedSource, FullSequenceSource (format-agnostic)
        pipeline.py      # GrainPipeline, create_grain_dls (HDF5 factory)
        stats.py         # compute_stats
    models/              # Neural network architectures (RNN, GRU, future: TCN, PINN)
    losses/              # Loss functions (normalized_mse, normalized_mae, rmse)
    training/            # Learner, factory functions (RNNLearner, GRULearner)
```

**Subpackage dependency layers:**
- **`_core`** (no internal deps): `Buffer` variable type
- **`data/`** (no cross-subpackage deps): `index` (Protocol) ← `hdf5_index`; `index` ← `sources` → `pipeline`; `stats`
- **`models/`** → `_core`
- **`losses/`** (no internal deps): pure loss functions
- **`training/`** → `models`, `data`, `losses`

## Environment

- **Package manager:** uv (not pip)
- **Python:** 3.12
- **JAX:** CPU-only on Apple Silicon (no GPU — jax-metal abandoned, jax-mps requires cp313)
- **Key deps:** flax (NNX API), optax, grain, h5py, numpy
- **CI:** GitHub Actions — lint (ruff) + test (pytest, Python 3.11 & 3.12)

## Code Style

- Type hints with modern union syntax (`str | None`)
- Flax NNX conventions: `nnx.Module`, `nnx.Param`, custom `Buffer(nnx.Variable)` for non-trainable state
- Line length: 100 (ruff config)
- Ruff rules: E, F, I, W

## Test Data

Tests use `test_data/WienerHammerstein/` (HDF5 files with `u` and `y` signal arrays in `train/`, `valid/`, `test/` subdirectories). The conftest creates a shared `pipeline` fixture with `win_sz=20, stp_sz=10, bs=4`.
