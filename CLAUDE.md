# CLAUDE.md

## Project Overview

tsjax is a JAX-based training library for time series system identification, extracted from [TSFast](https://github.com/daniel-at/tsfast) (PyTorch/fastai). It provides the same Learner factory pattern but uses Flax NNX models, Grain data loading, and Optax optimization.

**Relation to TSFast:** tsjax is a standalone sibling library — not a fork. It reimplements the training pipeline in JAX while maintaining numerical parity with TSFast (verified by 5-level validation). The two libraries cannot coexist in the same process due to a grain/sklearn mutex conflict.

## Package Structure

```
tsjax/
├── __init__.py        # Public API exports
├── __main__.py        # CLI entry: python -m tsjax.validation
├── hdf5_index.py      # HDF5MmapIndex: picklable mmap reader with h5py fallback
├── sources.py         # WindowedHDF5Source, FullSequenceSource (Grain data sources)
├── stats.py           # compute_norm_stats (exact match with TSFast)
├── pipeline.py        # create_grain_dls factory + GrainPipeline dataclass
├── models.py          # RNN/GRU: Flax NNX, raw-in/raw-out with internal normalization
├── train.py           # Loss functions: normalized_mse, normalized_mae, rmse
├── learner.py         # Learner class: fit(), fit_flat_cos()
├── factory.py         # RNNLearner(), GRULearner() factory functions
└── validation.py      # 5-level numeric validation vs TSFast (subprocess isolation)
```

## Commands

```bash
# Install in development mode
uv pip install -e ".[dev]"

# Run minimal example
python examples/00_minimal_example_jax.py

# Run validation against TSFast
python -m tsjax.validation --dataset test_data/WienerHammerstein --u u --y y --win_sz 100
python -m tsjax.validation --dataset test_data/pinn_var_ic --u u --y x v --win_sz 100
```

## Environment

- **Package manager:** uv (not pip)
- **Python:** 3.12
- **JAX:** 0.9.0.1, CPU-only (no GPU on Apple Silicon — see DESIGN.md)
- **Key deps:** flax, optax, grain, h5py, numpy

## Code Style

- Standard Python (not nbdev — unlike TSFast, source files are edited directly)
- Type hints with modern union syntax (`str | None`)
- Flax NNX conventions for models (`nnx.Module`, `nnx.Param`, `nnx.Variable`)
- Factory functions for Learner creation (mirrors TSFast's pattern)
