# Design Decisions & Architecture

## Architecture

### Data Pipeline: Raw Data, Model-Internal Normalization

The Grain pipeline (`create_grain_dls`) yields **raw physical values** — no normalization in the data pipeline. This differs from TSFast, where input `u` is normalized by a `Normalize` batch transform.

- Normalization stats (`u_mean/u_std/y_mean/y_std`) are computed from training data and stored on `GrainPipeline`
- The **model** normalizes input internally and denormalizes output (`Buffer` variables, excluded from gradients)
- Loss functions (`normalized_mse`, `normalized_mae`) compute in per-channel normalized space for balanced gradients

**Why:** Makes models self-contained (raw-in/raw-out), simplifies inference, and avoids TSFast's hidden type-dispatch normalization behavior.

### HDF5 mmap Bypass

`HDF5MmapIndex` extracts byte offsets from HDF5 files once at init (using h5py), then reads via `np.memmap` at runtime. This solves Grain's multiprocessing compatibility:

- **h5py:** Not picklable, not thread-safe — incompatible with Grain workers
- **np.memmap:** Trivially picklable (just path + offset), thread-safe, ~19% faster than h5py
- **Fallback:** Chunked/compressed datasets fall back to h5py (e.g., WienerHammerstein `.hdf5` files)
- **OS page cache** replaces TSFast's `Memoize` caching — automatic, zero code

**Requirement:** HDF5 datasets must use contiguous layout (h5py default, no `chunks=` argument) for mmap. TSFast's datasets already satisfy this.

### Computed Windowing

`WindowedHDF5Source` computes window boundaries on-the-fly from a flat index via `bisect`, storing only file paths + cumulative window counts. This replaces TSFast's `DfHDFCreateWindows` which precomputes 130K+ `(path, l_slc, r_slc)` tuples per file.

### Batch Convention

Grain yields `{'u': array, 'y': array}` dicts. The **loss function** unpacks them — the model accepts individual arrays. This follows the dominant JAX convention (Flax, Equinox, MaxText). `prediction=True` mode (concatenating `[u_norm, y_raw]`) is a single line in the loss function, not a pipeline transform.

---

## Key Design Decisions

### Why Grain over PyTorch DataLoader

- Pure JAX stack (no PyTorch dependency)
- Native deterministic reproducibility via seed propagation
- Built-in checkpointing (`get_state()`/`set_state()`) for mid-training resumption
- `.repeat()` + step-based iteration replaces TSFast's `NBatches_Factory` (simpler)
- Separate pipelines per split are cleaner than TSFast's combined-then-split via `ParentSplitter`

**Trade-off:** No equivalent to fastai's `TransformBlock` abstraction. TBPTT must be custom-built.

### Why Flax NNX (not Equinox)

- More built-in RNN cells (GRU, LSTM, SimpleCell, MGU, ConvLSTM)
- Native `nn.RNN` scan wrapper with `return_carry` and `seq_lengths`
- Native `padding='CAUSAL'` for TCN Conv1d
- Google-backed stability (vs Equinox's single maintainer)
- OOP API closer to PyTorch (easier migration from TSFast)

**Trade-off:** Equinox has a more cohesive PINN ecosystem (Diffrax by the same author). For future PINN work, consider Equinox or use Diffrax directly with Flax.

### Why Model-Internal Normalization

- Models are self-contained: raw physical values in, raw physical values out
- No need to track external normalization state during inference
- Norm stats stored as `Buffer` variables (excluded from gradients via Flax NNX)
- TSFast's approach (normalize in pipeline, denormalize in inference wrapper) requires navigating Learner internals

### Why `normalized_mse` Loss

Computes MSE in per-channel normalized space: `mean((pred - target)² / std²)`. This balances gradients across output channels with different physical scales (e.g., position in meters vs velocity in m/s).

---

## Known Issues

### Grain + sklearn/fastai Mutex Conflict

`grain._src.python.experimental.index_shuffle.python.index_shuffle_module.so` (abseil C++ threading) conflicts with sklearn's libomp. Since fastai imports sklearn:

- `torch + grain` = OK
- `sklearn + grain` = crash (`mutex lock failed: Invalid argument`)
- **fastai + grain = crash** → tsjax and TSFast cannot coexist in the same process

**Workaround:** `validation.py` computes TSFast reference values in a subprocess.

### JAX on Apple Silicon

- **jax-metal** (Apple): Abandoned, stuck on JAX 0.5.0, closed-source
- **jax-mps** (community): Early stage, only cp313 wheels (Feb 2026) — incompatible with cp312
- **Current setup:** JAX 0.9.0.1, Python 3.12, **CPU-only**

### HDF5 Chunked vs Contiguous

mmap bypass only works for contiguous HDF5 datasets. Chunked/compressed datasets fall back to h5py (slower, not multiprocessing-safe). The `pinn_var_ic` dataset (`.h5`) is contiguous; `WienerHammerstein` (`.hdf5`) is chunked.

---

## Ecosystem Choices

| Component | Library | Rationale |
|-----------|---------|-----------|
| NN modules | **Flax NNX** | Best RNN support, Google-backed, PyTorch-like OOP |
| Optimization | **Optax** | Composable chains replace TSFast callbacks (clipping, NaN, weight decay) |
| Data loading | **Grain** | Deterministic, picklable, checkpointable, no PyTorch dep |
| Logging | tqdm (current) | Minimal. wandb or TensorBoard planned |
| Checkpointing | Not yet | Orbax planned |

---

## Roadmap

### Implemented
- GRU/LSTM models with internal normalization (Flax NNX)
- Grain data pipeline with mmap HDF5 reading
- Learner with `fit()` and `fit_flat_cos()` schedules
- Factory functions: `RNNLearner()`, `GRULearner()`
- Loss functions: `normalized_mse`, `normalized_mae`, `rmse`
- 5-level validation against TSFast
- `n_skip` for RNN warmup timesteps

### Planned
- **TBPTT / stateful RNN** — hardest remaining feature; needs `IterDataset` with ordering guarantees
- **TCN** — dilated causal Conv1d blocks; Flax has native `padding='CAUSAL'`
- **CRNN** — TCN frontend + RNN backend composition
- **PIRNN** — biggest JAX win: `jax.grad` replaces all 12 finite-difference operators with exact autodiff
- **AR_Model** — autoregressive prediction with teacher forcing
- **More callbacks** — gradient clipping (via Optax chain), early stopping, LR finder
- **Orbax checkpointing** — async model/optimizer state saving
- **`prediction=True` mode** — one line in loss function (concat `[u_norm, y_raw]`)
- **Noise/bias injection** — `jax.random.normal` transforms
- **InferenceWrapper** — simpler than TSFast's (no fastai internals to navigate)

---

## TSFast Comparison

### Normalization Differences

| Aspect | TSFast | tsjax |
|--------|--------|-------|
| Pipeline output | Input `u` normalized, output `y` raw | Both raw |
| Where normalization happens | `Normalize` batch transform (type-dispatched) | Inside model (`Buffer` variables) |
| Model I/O | Normalized input in, raw output out | Raw in, raw out |
| Loss computation | Raw space | Normalized space (per-channel) |
| Inference | Requires denormalization wrapper | Model is self-contained |

### Architecture Mapping

| TSFast | tsjax | Notes |
|--------|-------|-------|
| `create_dls()` | `create_grain_dls()` | Returns `GrainPipeline` instead of fastai `DataLoaders` |
| `RNNLearner()` | `RNNLearner()` | Same factory pattern, different internals |
| `Learner.fit_flat_cos()` | `Learner.fit_flat_cos()` | Same API, Optax schedule internally |
| `HDF2Sequence` | `HDF5MmapIndex` + `WindowedHDF5Source` | mmap instead of h5py |
| `Normalize` batch transform | Model-internal normalization | `Buffer` variables |
| `TbpttDl` + `TbpttResetCB` | Not yet implemented | Planned |
| `GradientClipping` callback | `optax.clip_by_global_norm` in chain | Optimizer composition |
| `SkipNaNCallback` | `optax.zero_nans()` in chain | Optimizer composition |
| Finite-difference operators (12) | `jax.grad` | Exact autodiff |
| `InferenceWrapper` | Unnecessary (model is raw-in/raw-out) | Simpler |
