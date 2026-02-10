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
                    ┌────────────────┼────────────────┐
                    ▼                                  ▼
 ┌──────────────────────────────┐   ┌──────────────────────────────────┐
 │  STAGE 2a: FILE INDEXING     │   │  STAGE 2b: NORM STATS            │
 │                              │   │                                  │
 │  HDF5MmapIndex               │   │  compute_norm_stats()            │
 │  ├── scans HDF5 at init      │   │  ├── reads all training files    │
 │  ├── stores byte offsets     │   │  ├── accumulates mean/std        │
 │  └── reads via np.memmap     │   │  └── returns (mean, std) float32 │
 │      (fallback: h5py if      │   │                                  │
 │       chunked/compressed)    │   │  Output:                         │
 │                              │   │    u_mean, u_std                  │
 │  One index per split:        │   │    y_mean, y_std                  │
 │    train_index               │   │                                  │
 │    valid_index               │   │  Swappable: stats algorithm       │
 │    test_index                │   │  Future:    per-file stats,       │
 │                              │   │             running stats,        │
 │  Swappable: whole class      │   │             cached stats          │
 │  Future:    CSVIndex,        │   │                                  │
 │             NetCDFIndex      │   │                                  │
 └──────────────┬───────────────┘   └─────────────────┬────────────────┘
                │                                      │
                ▼                                      │
 ┌──────────────────────────────────┐                  │
 │  STAGE 3: DATA SOURCES           │                  │
 │                                  │                  │
 │  WindowedSource (train/valid)    │                  │
 │  ├── __len__: total windows      │                  │
 │  ├── __getitem__(idx):           │                  │
 │  │   bisect → file + offset      │                  │
 │  │   index.read_signals()        │                  │
 │  │   → {"u": array, "y": array}  │                  │
 │  └── params: win_sz, stp_sz      │                  │
 │                                  │                  │
 │  FullSequenceSource (test)       │                  │
 │  ├── __len__: num files          │                  │
 │  └── __getitem__: full sequence  │                  │
 │                                  │                  │
 │  Swappable: any class with       │                  │
 │    __len__ + __getitem__          │                  │
 │  Future: transforms/augmentation │                  │
 │    inserted here (noise, bias,   │                  │
 │    slicing, resampling)          │                  │
 └──────────────┬───────────────────┘                  │
                │                                      │
                ▼                                      │
 ┌──────────────────────────────────┐                  │
 │  STAGE 4: GRAIN PIPELINE         │                  │
 │                                  │                  │
 │  grain.MapDataset.source(src)    │                  │
 │    .shuffle(seed=42)   # train   │                  │
 │    .batch(bs, drop_remainder)    │                  │
 │                                  │                  │
 │  Yields: {"u": (bs,win,n_u),     │                  │
 │           "y": (bs,win,n_y)}     │                  │
 │                                  │                  │
 │  Swappable: batch size, seed,    │                  │
 │    shuffle strategy              │                  │
 │  Future:    weighted sampling,   │                  │
 │    TBPTT loader, batch limiting  │                  │
 └──────────────┬───────────────────┘                  │
                │                                      │
                ▼                                      │
 ┌─────────────────────────────────────────────────────┼───────────────┐
 │  GrainPipeline (dataclass)                          │               │
 │  ├── .train  (MapDataset)                           │               │
 │  ├── .valid  (MapDataset)          ◄────────────────┘               │
 │  ├── .test   (MapDataset)     norm stats stored here                │
 │  ├── .u_mean, .u_std          (passed to model + loss)              │
 │  └── .y_mean, .y_std                                                │
 └───────────────────────────────────┬─────────────────────────────────┘
                                     │
              ┌──────────────────────┼──────────────────────┐
              ▼                      ▼                      ▼
 ┌────────────────────┐ ┌────────────────────┐ ┌────────────────────────┐
 │  MODEL CREATION     │ │  LOSS FUNCTION      │ │  METRICS               │
 │                     │ │                     │ │                        │
 │  create_rnn()       │ │  normalized_mae()   │ │  rmse() etc.           │
 │  ├── infers I/O     │ │  normalized_mse()   │ │                        │
 │  │   sizes from     │ │                     │ │  Signature:            │
 │  │   pipeline stats │ │  Signature:         │ │  fn(pred, target,      │
 │  ├── creates RNN    │ │  fn(pred, target,   │ │     y_mean, y_std)     │
 │  │   with Buffer    │ │     y_mean, y_std)  │ │     → scalar           │
 │  │   norm stats     │ │     → scalar        │ │                        │
 │  └── returns model  │ │                     │ │  Swappable: any fn     │
 │                     │ │  Normalizes both     │ │    with same signature │
 │  Swappable:         │ │  pred and target     │ │  Future: NRMSE, VAF,   │
 │  rnn_type=gru|lstm  │ │  internally          │ │    cosine similarity   │
 │  hidden_size        │ │                     │ └────────────────────────┘
 │  num_layers         │ │  Swappable: any fn  │
 │                     │ │    with same sig     │
 │  Future: TCN, CRNN, │ │  Future: skip-N,    │
 │  PIRNN, FranSys     │ │    weighted, NaN    │
 └─────────┬───────────┘ └─────────┬───────────┘
           │                       │
           ▼                       ▼
 ┌─────────────────────────────────────────────────────────────────────┐
 │  STAGE 5: MODEL (forward pass)                                      │
 │                                                                     │
 │  RNN(nnx.Module)                                                    │
 │  ├── Buffer: u_mean, u_std, y_mean, y_std  (frozen, no gradients)  │
 │  ├── nnx.RNN layers (GRU or LSTM cells)                            │
 │  └── nnx.Linear output projection                                   │
 │                                                                     │
 │  __call__(x):   x is raw (bs, seq_len, n_input)                    │
 │    1. normalize:    x = (x - u_mean) / u_std                       │
 │    2. RNN layers:   x = rnn_layer_1(x) → ... → rnn_layer_N(x)     │
 │    3. linear proj:  x = linear(x)                                   │
 │    4. denormalize:  x = x * y_std + y_mean                         │
 │    → returns raw (bs, seq_len, n_output)                            │
 │                                                                     │
 │  Self-contained: raw in → raw out. No external normalization.       │
 └───────────────────────────────────┬─────────────────────────────────┘
                                     │
                                     ▼
 ┌─────────────────────────────────────────────────────────────────────┐
 │  STAGE 6: TRAINING LOOP  (Learner._fit)                             │
 │                                                                     │
 │  Per epoch:                                                         │
 │  ┌───────────────────────────────────────────────────────────────┐  │
 │  │  TRAIN                                                        │  │
 │  │  for batch in pipeline.train:                                 │  │
 │  │    u, y = batch["u"], batch["y"]          # raw arrays        │  │
 │  │    pred = model(u)                        # raw → raw         │  │
 │  │    loss = loss_func(pred, y, y_mean, y_std)  # normalized     │  │
 │  │    grads = jax.grad(loss)                                     │  │
 │  │    optimizer.update(model, grads)          # optax step       │  │
 │  │    ────────────────────────────────────                       │  │
 │  │    Optional: n_skip truncates first N timesteps from loss     │  │
 │  │    Future:   callbacks fire here (grad clip, NaN skip, etc.)  │  │
 │  └───────────────────────────────────────────────────────────────┘  │
 │  ┌───────────────────────────────────────────────────────────────┐  │
 │  │  VALIDATE                                                     │  │
 │  │  for batch in pipeline.valid:                                 │  │
 │  │    pred = model(u)                                            │  │
 │  │    loss = loss_func(pred, y, y_mean, y_std)                   │  │
 │  │    metrics = [m(pred, y, y_mean, y_std) for m in metric_fns]  │  │
 │  └───────────────────────────────────────────────────────────────┘  │
 │                                                                     │
 │  Optimizer: optax.adam(lr) or optax.adam(schedule)                   │
 │  Swappable: optimizer (any optax transform), lr schedule            │
 │  Future:    optax.chain(clip, adam, ema) for gradient callbacks,     │
 │             early stopping, checkpointing, LR finder                │
 └───────────────────────────────────┬─────────────────────────────────┘
                                     │
                                     ▼
 ┌─────────────────────────────────────────────────────────────────────┐
 │  OUTPUT: trained model                                              │
 │                                                                     │
 │  learner.model     ← trained RNN with updated nnx.Param weights    │
 │  learner.train_losses, learner.valid_losses, learner.valid_metrics  │
 │                                                                     │
 │  Inference: model(raw_input) → raw_output  (no wrapper needed)      │
 │  Future:    InferenceWrapper, Orbax checkpointing, export           │
 └─────────────────────────────────────────────────────────────────────┘
```

## Swappability Summary

| Component | Interface / Contract | Swap by |
|---|---|---|
| File reader | `SignalIndex` protocol: `.paths`, `.read_signals()`, `.get_seq_len()` | New class implementing `SignalIndex` (e.g. `CSVIndex`) |
| Data source | `__len__()` + `__getitem__(idx)` → `{"u": ndarray, "y": ndarray}` | New class |
| Norm stats | `fn(files, signals) → (mean, std)` | New function |
| Model | `nnx.Module` with `__call__`: `(bs, seq, n_in) → (bs, seq, n_out)` | New class (TCN, CRNN, PIRNN) |
| Loss function | `fn(pred, target, y_mean, y_std) → scalar` | Any function with same signature |
| Metrics | Same signature as loss | Any function with same signature |
| Optimizer | Any `optax.GradientTransformation` | `optax.*` |
| LR schedule | Optax schedule passed to optimizer | `optax.*` |
| Data augmentation (future) | Inserts between `Source.__getitem__` and Grain | Transform on source |
| Callbacks (future) | Hooks in `_fit` loop or `optax.chain()` for grad ops | Callback classes or optax chains |

## Key Contracts

Everything flows through **two contracts**:

1. **Data contract**: `{"u": (bs, seq, n_in), "y": (bs, seq, n_out)}` — dict of raw numpy/jax arrays
2. **Loss/metric contract**: `fn(pred, target, y_mean, y_std) → scalar`

Any component that respects these can be swapped without touching the rest.

## File Mapping

| Stage | File | Key exports |
|---|---|---|
| File discovery | `data/pipeline.py` | `_get_split_files()` |
| File indexing | `data/hdf5_index.py` | `HDF5MmapIndex`, `SignalInfo` |
| Norm stats | `data/stats.py` | `compute_norm_stats()` |
| Signal index protocol | `data/index.py` | `SignalIndex` |
| Data sources | `data/sources.py` | `WindowedSource`, `FullSequenceSource` |
| Pipeline assembly | `data/pipeline.py` | `GrainPipeline`, `create_grain_dls()` |
| Model | `models/rnn.py` | `RNN`, `GRU` |
| Loss functions | `losses/core.py` | `normalized_mse`, `normalized_mae`, `rmse` |
| Training loop | `training/learner.py` | `Learner` |
| Factory | `training/factory.py` | `RNNLearner`, `GRULearner`, `create_rnn`, `create_gru` |
| Framework types | `_core.py` | `Buffer` |
