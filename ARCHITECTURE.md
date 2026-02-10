# Architecture & Component Flowchart

From raw files to trained model — what each component does, how they connect, and what's swappable.

## Pipeline Flowchart

```
                        TSJAX COMPONENT FLOWCHART
                    From raw files to trained model

 ┌─────────────────────────────────────────────────────────────────────┐
 │  STAGE 1: FILE DISCOVERY                                           │
 │                                                                     │
 │  dataset/                                                           │
 │    ├── train/*.hdf5          _get_split_files()                     │
 │    ├── valid/*.hdf5     ───► returns list[str] of paths per split   │
 │    └── test/*.hdf5                                                  │
 │                                                                     │
 │  Swappable: directory layout, file extensions (.hdf5/.h5)           │
 │  Future:    CSV discovery, NetCDF discovery, custom splitters       │
 └───────────────────────────────────┬─────────────────────────────────┘
                                     │
                                     ▼
 ┌─────────────────────────────────────────────────────────────────────┐
 │  STAGE 2: SIGNAL STORE                                              │
 │                                                                     │
 │  HDF5Store                                                          │
 │  ├── scans HDF5 at init (extracts mmap offsets, shapes, dtypes)    │
 │  ├── reads via np.memmap at runtime (fallback: h5py if compressed) │
 │  ├── fully picklable (Grain multiprocessing safe)                  │
 │  └── optional: preload=True caches full arrays in memory           │
 │                                                                     │
 │  One store per split: train_store, valid_store, test_store          │
 │                                                                     │
 │  ┌──────────────────────────────────────────────┐                  │
 │  │  Optional: ResampledStore wrapper             │                  │
 │  │  ├── wraps any SignalStore                    │                  │
 │  │  ├── resamples lazily on first access         │                  │
 │  │  ├── caches resampled arrays (transient)     │                  │
 │  │  ├── factor: uniform or per-file callable    │                  │
 │  │  └── resample_fn: resample_interp (default)  │                  │
 │  │       or resample_fft                         │                  │
 │  └──────────────────────────────────────────────┘                  │
 │                                                                     │
 │  Protocol: SignalStore                                              │
 │    .paths: Sequence[str]                                            │
 │    .get_seq_len(path, signal?) -> int                               │
 │    .read_signals(path, signals, l_slc, r_slc) -> ndarray           │
 │                                                                     │
 │  Swappable: whole class (any SignalStore implementation)            │
 │  Future:    CSVStore, ParquetStore, NetCDFStore                    │
 └───────────────────────────────────┬─────────────────────────────────┘
                                     │
                    ┌────────────────┼────────────────┐
                    ▼                                  ▼
 ┌──────────────────────────────┐   ┌──────────────────────────────────┐
 │  STAGE 3a: READERS           │   │  STAGE 3b: NORM STATS            │
 │                              │   │                                  │
 │  SequenceReader              │   │  compute_stats(source, key)      │
 │  ├── reads signal windows   │   │  ├── dispatches on reader type   │
 │  │   from a SignalStore      │   │  │   (Sequence/Scalar/Feature)   │
 │  ├── windowed or full-seq   │   │  ├── optional transform kwarg    │
 │  │   (l_slc>=r_slc → full)  │   │  │   for per-sample transforms   │
 │  └── (seq_len, n_signals)    │   │  └── returns NormStats(mean,std) │
 │                              │   │                                  │
 │  ScalarAttrReader            │   │  Output per key:                 │
 │  ├── reads per-file HDF5    │   │    NormStats(mean, std)           │
 │  │   root-level attributes   │   │                                  │
 │  ├── pre-caches at init     │   │  Swappable: stats algorithm      │
 │  └── returns (n_attrs,)      │   │  Future:    cached stats,        │
 │                              │   │             running stats         │
 │  FeatureReader               │   │                                  │
 │  ├── reads window from      │   │                                  │
 │  │   store, applies fn       │   │                                  │
 │  └── returns (n_features,)   │   │                                  │
 │                              │   │                                  │
 │  Protocol: Reader            │   │                                  │
 │    .signals: list[str]       │   │                                  │
 │    .__call__(path, l, r)     │   │                                  │
 │      -> ndarray              │   │                                  │
 │                              │   │                                  │
 │  Swappable: any Reader impl  │   │                                  │
 └──────────────┬───────────────┘   └─────────────────┬────────────────┘
                │                                      │
                ▼                                      │
 ┌──────────────────────────────────┐                  │
 │  STAGE 4: DATA SOURCE            │                  │
 │                                  │                  │
 │  DataSource                      │                  │
 │  ├── store + dict of Readers     │                  │
 │  ├── win_sz=None → _FileIndex   │                  │
 │  │   (one sample per file)       │                  │
 │  ├── win_sz=int → _WindowIndex  │                  │
 │  │   (sliding windows, bisect)   │                  │
 │  ├── __len__: total samples      │                  │
 │  └── __getitem__(idx):           │                  │
 │      resolve(idx) → path,l,r     │                  │
 │      {key: reader(path,l,r)}     │                  │
 │      → dict[str, ndarray]        │                  │
 │                                  │                  │
 │  Swappable: any class with       │                  │
 │    __len__ + __getitem__          │                  │
 │  Future: augmentation transforms │                  │
 └──────────────┬───────────────────┘                  │
                │                                      │
                ▼                                      │
 ┌──────────────────────────────────┐                  │
 │  STAGE 5: GRAIN PIPELINE         │                  │
 │                                  │                  │
 │  grain.MapDataset.source(src)    │                  │
 │    .map(transforms)   # optional │                  │
 │    .shuffle(seed=42)  # train    │                  │
 │    .batch(bs, drop_remainder)    │                  │
 │                                  │                  │
 │  Yields: dict[str, ndarray]      │                  │
 │  e.g. {"u": (bs,win,n_u),        │                  │
 │        "y": (bs,win,n_y)}        │                  │
 │  or   {"u": (bs,seq,n_u),        │                  │
 │        "y": (bs,n_attrs)}         │                  │
 │                                  │                  │
 │  Swappable: batch size, seed,    │                  │
 │    shuffle strategy, transforms  │                  │
 │  Future:    weighted sampling,   │                  │
 │    TBPTT loader, batch limiting  │                  │
 └──────────────┬───────────────────┘                  │
                │                                      │
                ▼                                      │
 ┌─────────────────────────────────────────────────────┼───────────────┐
 │  GrainPipeline (dataclass)                          │               │
 │  ├── .train  (MapDataset)                           │               │
 │  ├── .valid  (MapDataset)          ◄────────────────┘               │
 │  ├── .test   (MapDataset)     stats stored here                     │
 │  ├── .stats: dict[str, NormStats]  (passed to model + loss)        │
 │  ├── .input_keys, .target_keys                                      │
 │  └── .train_source, .valid_source, .test_source  (raw DataSources) │
 └───────────────────────────────────┬─────────────────────────────────┘
                                     │
              ┌──────────────────────┼──────────────────────┐
              ▼                      ▼                      ▼
 ┌────────────────────┐ ┌────────────────────┐ ┌────────────────────────┐
 │  MODEL CREATION     │ │  LOSS FUNCTION      │ │  METRICS               │
 │                     │ │                     │ │                        │
 │  create_rnn()       │ │  normalized_mae()   │ │  rmse() etc.           │
 │  ├── infers I/O     │ │  normalized_mse()   │ │                        │
 │  │   sizes from     │ │  cross_entropy_loss()│ │  Signature:            │
 │  │   pipeline stats │ │                     │ │  fn(pred, target,      │
 │  ├── creates model  │ │  Signature:         │ │     y_mean, y_std)     │
 │  │   with Buffer    │ │  fn(pred, target,   │ │     → scalar           │
 │  │   norm stats     │ │     y_mean, y_std)  │ │                        │
 │  └── returns model  │ │     → scalar        │ │  Swappable: any fn     │
 │                     │ │                     │ │    with same signature │
 │  Swappable:         │ │  Regression losses  │ │  Future: NRMSE, VAF,   │
 │  rnn_type=gru|lstm  │ │  normalize both     │ │    cosine similarity   │
 │  hidden_size        │ │  pred and target    │ └────────────────────────┘
 │  num_layers         │ │  per-channel        │
 │                     │ │                     │
 │  Future: TCN, CRNN, │ │  Swappable: any fn  │
 │  PIRNN, FranSys     │ │    with same sig     │
 └─────────┬───────────┘ └─────────┬───────────┘
           │                       │
           ▼                       ▼
 ┌─────────────────────────────────────────────────────────────────────┐
 │  STAGE 6: MODELS (forward pass)                                     │
 │                                                                     │
 │  RNN/GRU (nnx.Module) — sequence → sequence                        │
 │  ├── Buffer: u_mean, u_std, y_mean, y_std  (frozen, no gradients)  │
 │  ├── nnx.RNN layers (GRU or LSTM cells)                            │
 │  ├── nnx.Linear output projection                                   │
 │  └── __call__(u): (bs, seq_len, n_in) → (bs, seq_len, n_out)      │
 │      1. normalize:    x = (u - u_mean) / u_std                     │
 │      2. RNN layers:   x = rnn_layer_1(x) → ... → rnn_layer_N(x)   │
 │      3. linear proj:  x = linear(x)                                 │
 │      4. denormalize:  x = x * y_std + y_mean                       │
 │                                                                     │
 │  RNNEncoder (nnx.Module) — sequence → scalar                       │
 │  ├── same Buffer + RNN layers as RNN                                │
 │  ├── last-hidden-state pooling: x[:, -1, :]                        │
 │  └── __call__(u): (bs, seq_len, n_in) → (bs, n_out)               │
 │      For classification: y_mean/y_std left as identity (raw logits) │
 │      For regression: y_mean/y_std denormalize to physical units     │
 │                                                                     │
 │  MLP (nnx.Module) — scalar → scalar                                │
 │  ├── same Buffer normalization pattern                              │
 │  ├── hidden layers with ReLU (configurable sizes, default [64,32]) │
 │  └── __call__(u): (bs, n_in) → (bs, n_out)                        │
 │                                                                     │
 │  All models: raw in → raw out.  No external normalization needed.  │
 └───────────────────────────────────┬─────────────────────────────────┘
                                     │
                                     ▼
 ┌─────────────────────────────────────────────────────────────────────┐
 │  STAGE 7: TRAINING LOOP  (Learner._fit)                             │
 │                                                                     │
 │  Per epoch:                                                         │
 │  ┌───────────────────────────────────────────────────────────────┐  │
 │  │  TRAIN                                                        │  │
 │  │  for batch in pipeline.train:                                 │  │
 │  │    inputs = {k: batch[k] for k in input_keys}                │  │
 │  │    y = batch[target_key]                                      │  │
 │  │    pred = model(**inputs)                   # raw → raw       │  │
 │  │    loss = loss_func(pred, y, y_mean, y_std) # normalized      │  │
 │  │    grads = nnx.value_and_grad(loss_fn)(model, inputs, y)     │  │
 │  │    optimizer.update(model, grads)            # optax step     │  │
 │  │    ────────────────────────────────────                       │  │
 │  │    Optional: n_skip truncates first N timesteps from loss     │  │
 │  │    Future:   callbacks fire here (grad clip, NaN skip, etc.)  │  │
 │  └───────────────────────────────────────────────────────────────┘  │
 │  ┌───────────────────────────────────────────────────────────────┐  │
 │  │  VALIDATE                                                     │  │
 │  │  for batch in pipeline.valid:                                 │  │
 │  │    pred = model(**inputs)                                     │  │
 │  │    loss = loss_func(pred, y, y_mean, y_std)                   │  │
 │  │    metrics = [m(pred, y, y_mean, y_std) for m in metric_fns]  │  │
 │  └───────────────────────────────────────────────────────────────┘  │
 │                                                                     │
 │  JIT-compiled: train_step and eval_step via @nnx.jit               │
 │  Optimizer: optax.adam(lr) or optax.adam(schedule)                   │
 │  Schedules: fit() = constant LR, fit_flat_cos() = flat + cosine    │
 │  Swappable: optimizer (any optax transform), lr schedule            │
 │  Future:    optax.chain(clip, adam, ema) for gradient callbacks,     │
 │             early stopping, checkpointing, LR finder                │
 └───────────────────────────────────┬─────────────────────────────────┘
                                     │
                                     ▼
 ┌─────────────────────────────────────────────────────────────────────┐
 │  OUTPUT: trained model                                              │
 │                                                                     │
 │  learner.model     ← trained module with updated nnx.Param weights │
 │  learner.train_losses, learner.valid_losses, learner.valid_metrics  │
 │                                                                     │
 │  Inference: model(raw_input) → raw_output  (no wrapper needed)      │
 │  Visualization: learner.show_batch(), learner.show_results()        │
 │  Future:    InferenceWrapper, Orbax checkpointing, export           │
 └─────────────────────────────────────────────────────────────────────┘
```

## Swappability Summary

| Component | Interface / Contract | Swap by |
|---|---|---|
| Signal store | `SignalStore` protocol: `.paths`, `.read_signals()`, `.get_seq_len()` | New class implementing `SignalStore` (e.g. `CSVStore`) |
| Resampling | `ResampledStore` wraps any `SignalStore`; `ResampleFn` signature | Custom `resample_fn` or new store wrapper |
| Reader | `Reader` protocol: `.signals`, `.__call__(path, l_slc, r_slc)` → `ndarray` | New class implementing `Reader` |
| Data source | `__len__()` + `__getitem__(idx)` → `dict[str, ndarray]` | New class |
| Transforms | `Transform` = `Callable[[ndarray], ndarray]` per batch key | Any function with same signature |
| Norm stats | `compute_stats(source, key) → NormStats` | Implement `compute_stats() → NormStats` method on reader |
| Model | `nnx.Module` with `__call__`: input arrays → output array | New class (TCN, CRNN, PIRNN) |
| Loss function | `fn(pred, target, y_mean, y_std) → scalar` | Any function with same signature |
| Metrics | Same signature as loss | Any function with same signature |
| Optimizer | Any `optax.GradientTransformation` | `optax.*` |
| LR schedule | Optax schedule passed to optimizer | `optax.*` |

## Key Contracts

Everything flows through **three contracts**:

1. **Data contract**: `dict[str, ndarray]` — batch dicts with arbitrary key names (e.g. `{"u": (bs, seq, n_in), "y": (bs, seq, n_out)}` for simulation, or `{"u": (bs, seq, n_in), "y": (bs, n_classes)}` for classification)
2. **Loss/metric contract**: `fn(pred, target, y_mean, y_std) → scalar`
3. **Reader contract**: `Reader` protocol — `signals: list[str]`, `__call__(path, l_slc, r_slc) → ndarray`

Any component that respects these can be swapped without touching the rest.

## Factory Entry Points

| Use case | Factory | Model | Loss |
|---|---|---|---|
| Time series simulation | `RNNLearner` / `GRULearner` | `RNN` | `normalized_mae` |
| Sequence classification | `ClassifierLearner` | `RNNEncoder` | `cross_entropy_loss` |
| Tabular regression | `RegressionLearner` | `MLP` | `normalized_mse` |
| Benchmark suite | `create_grain_dls_from_spec` | (any of above) | (any of above) |

## File Mapping

| Stage | File | Key exports |
|---|---|---|
| File discovery | `data/pipeline.py` | `_get_split_files()` |
| Signal store protocol | `data/store.py` | `SignalStore` |
| HDF5 store | `data/hdf5_store.py` | `HDF5Store`, `SignalInfo`, `read_hdf5_attr()` |
| Resampling | `data/resample.py` | `ResampledStore`, `resample_interp()`, `resample_fft()` |
| Readers + data source | `data/sources.py` | `DataSource`, `SequenceReader`, `ScalarAttrReader`, `FeatureReader`, `Reader`, `ScalarAttr`, `Feature` |
| Norm stats | `data/stats.py` | `NormStats`, `compute_stats()` |
| Transforms | `data/item_transforms.py` | `Transform`, `stft_transform()` |
| Pipeline assembly | `data/pipeline.py` | `GrainPipeline`, `create_grain_dls()`, `create_simulation_dls()` |
| Benchmark integration | `data/benchmark.py` | `create_grain_dls_from_spec()`, `BENCHMARK_DL_KWARGS` |
| RNN model | `models/rnn.py` | `RNN`, `GRU` |
| RNN encoder | `models/encoder.py` | `RNNEncoder` |
| MLP model | `models/mlp.py` | `MLP` |
| Regression losses | `losses/core.py` | `normalized_mse`, `normalized_mae`, `rmse` |
| Classification loss | `losses/classification.py` | `cross_entropy_loss` |
| Training loop | `training/learner.py` | `Learner` |
| Factory functions | `training/factory.py` | `RNNLearner`, `GRULearner`, `ClassifierLearner`, `RegressionLearner`, `create_rnn`, `create_gru` |
| Visualization | `viz.py` | `plot_batch`, `plot_results`, `plot_scalar_batch`, `plot_classification_results`, `plot_regression_scatter` |
| Framework types | `_core.py` | `Buffer` |
