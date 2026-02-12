# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

tsjax is a JAX-based training library for time series system identification, a standalone sibling of [TSFast](https://github.com/daniel-at/tsfast) (PyTorch/fastai). It reimplements the training pipeline using Flax NNX models, Grain data loading, and Optax optimization. The two libraries cannot coexist in the same process (grain/sklearn mutex conflict).

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
