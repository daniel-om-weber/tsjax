# tsjax Architecture Plan: Full TSFast Feature Parity

## Context

tsjax handles basic RNN training for time series system identification. TSFast has many more features (PINNs, autoregressive models, TCN/CRNN, spectrograms, quaternion augmentation, HPO, TBPTT, weighted sampling, etc.) that are deeply coupled through fastai's callback system and implicit type dispatch. This plan designs tsjax's architecture so all those features can be added incrementally with **explicit, minimal interconnections** — no monolithic callback system, no implicit behavior.

## Core Principle: Four Explicit Mechanisms Replace Callbacks

Every TSFast callback maps to exactly one of these:

| Mechanism | Scope | Replaces |
|---|---|---|
| **optax chains** | Gradient-level | GradientClipping, SkipNaN, zero_nans |
| **Loss composition** | Loss-level | PhysicsLoss, FranSys, AuxiliaryOutput, SkipNLoss |
| **batch_transform / model-internal** | Batch-level | TbpttReset, PredictionCB, VarySeqLen, noise injection |
| **fit_iter() generator** | Epoch-level | Ray Tune reporting, early stopping, checkpointing |

---

## Target Package Structure

```
tsjax/
    _core.py                    # Buffer(nnx.Variable) — unchanged
    data/
        store.py                # SignalStore Protocol — unchanged
        hdf5_store.py           # HDF5Store — unchanged
        resample.py             # ResampledStore — unchanged
        sources.py              # WindowedSource, FullSequenceSource — unchanged
        pipeline.py             # GrainPipeline, create_grain_dls — unchanged
        stats.py                # compute_norm_stats — unchanged
        sampling.py             # NEW: weighted sampling, n_batches control
    models/
        rnn.py                  # RNN/GRU — unchanged
        tcn.py                  # NEW: TCN (dilated causal conv1d)
        crnn.py                 # NEW: CRNN (TCN frontend + RNN backend)
        pirnn.py                # NEW: PIRNN (dual encoders, physics-informed)
        ar.py                   # NEW: autoregressive models (ARProg, FranSys)
    losses/
        core.py                 # normalized_mse, normalized_mae, rmse — unchanged
        compose.py              # NEW: compose_losses(), skip_n(), masked_loss()
        physics.py              # NEW: physics loss terms (uses jax.grad, not finite diff)
        quaternion.py           # NEW: quaternion-specific losses
    training/
        learner.py              # Learner — add fit_iter(), batch_transform param
        factory.py              # RNNLearner, GRULearner — add TCN/CRNN/PIRNN factories
    transforms/                 # NEW subpackage
        noise.py                # noise/bias injection (jax.random based)
        quaternion.py           # QuaternionAugmentation (rotation groups)
        truncate.py             # sequence truncation schedules
    inference.py                # NEW: minimal (model is already raw-in/raw-out)
    tune.py                     # NEW: Ray Tune integration (HPOptimizer)
    viz.py                      # plot_batch, plot_results — extend for new types
```

---

## Changes by Area

### 1. Training Loop (`training/learner.py`)

**Add `fit_iter()` generator** — yields control after each epoch:

```python
def fit_iter(self, n_epochs, lr=1e-3, optimizer=None, schedule_fn=None):
    """Generator: yields EpochResult after each epoch."""
    # setup optimizer, JIT compile steps (same as _fit)
    for epoch in range(n_epochs):
        train_loss = self._run_epoch(train=True, ...)
        valid_loss, metrics = self._run_epoch(train=False, ...)
        yield EpochResult(epoch, train_loss, valid_loss, metrics)
```

Caller usage for Ray Tune:
```python
for result in learner.fit_iter(100, lr=lr):
    ray.tune.report(valid_loss=result.valid_loss)
    if result.valid_loss < threshold:
        break
```

**Add `batch_transform` parameter** to Learner:

```python
@dataclass
class Learner:
    batch_transform: Callable | None = None  # applied to each batch before forward pass
```

In `_train_step`: if `batch_transform` is not None, apply it to the batch dict before unpacking. This handles noise injection, sequence truncation, etc.

**Refactor `_fit` to use `fit_iter` internally** — `fit()` and `fit_flat_cos()` become thin wrappers that consume the generator.

### 2. Loss Composition (`losses/compose.py`)

```python
def compose_losses(*weighted_losses: tuple[LossFn, float]) -> LossFn:
    """Combine multiple loss functions with weights.

    Usage: compose_losses((normalized_mse, 1.0), (physics_loss, 0.1))
    """

def skip_n(loss_fn: LossFn, n: int) -> LossFn:
    """Wrap loss to skip first n timesteps (RNN warmup)."""

def masked_loss(loss_fn: LossFn, channel_mask: Array) -> LossFn:
    """Wrap loss to apply only to selected output channels (PIRNN supervised subset)."""
```

All loss functions keep the existing signature: `fn(pred, target, y_mean, y_std) → scalar`. Composition returns a function with the same signature.

### 3. Models

**TCN** (`models/tcn.py`):
- Dilated causal Conv1d blocks using Flax's native `padding='CAUSAL'`
- Same interface: `__call__(x) → (bs, seq, n_out)`, raw-in/raw-out with Buffer norm stats

**CRNN** (`models/crnn.py`):
- TCN frontend → RNN backend, composed from TCN and RNN modules
- Same interface as RNN

**PIRNN** (`models/pirnn.py`):
- Dual encoders: sequence encoder (RNN on u,y) + state encoder (linear on physical state)
- `n_y_supervised` param for partial supervision
- Physics loss is a **separate loss function** in `losses/physics.py`, not a callback
- `jax.grad` replaces TSFast's 12 finite-difference operators (exact autodiff)

**AR / FranSys** (`models/ar.py`):
- Autoregressive prediction: model concatenates own output to input internally
- FranSys: diagnosis + prognosis modules with shared final layer
- Hidden state sync between diagnosis/prognosis is model-internal logic
- Multiple FranSys loss terms → `compose_losses()` with named terms

### 4. Data: Sampling Control (`data/sampling.py`)

```python
def weighted_source(source, weights: np.ndarray) -> grain.MapDataset:
    """Wraps a source with weighted sampling."""

def limit_batches(dataset: grain.MapDataset, max_batches: int) -> grain.MapDataset:
    """Limits batches per epoch (replaces BatchLimit_Factory)."""
```

These are Grain-native: `.shuffle()` with custom sampler, `.slice()` for limiting.

### 5. Transforms (`transforms/`)

Pure functions that take a batch dict and return a modified batch dict. Used via `batch_transform`:

```python
# noise.py
def add_noise(key, std=0.01):
    def transform(batch):
        return {**batch, "u": batch["u"] + std * jax.random.normal(key, batch["u"].shape)}
    return transform

# truncate.py
def truncate_schedule(epoch, total_epochs, min_len, max_len):
    """Returns a batch_transform that truncates sequences based on epoch progress."""

# quaternion.py
def quaternion_augmentation(key, groups):
    """Rotates quaternion/vector signal groups by random rotation."""
```

### 6. Inference (`inference.py`)

Minimal because models are already raw-in/raw-out:

```python
def predict(model, data: np.ndarray) -> np.ndarray:
    """Run inference on raw input array. Handles batching and JAX conversion."""
```

For stateful models (TBPTT, AR), add `predict_stateful()` that manages hidden state across chunks.

### 7. HPO / Ray Tune (`tune.py`)

```python
class HPOptimizer:
    def __init__(self, create_learner_fn, pipeline):
        ...

    def optimize(self, config, n_epochs=100, **tune_kwargs):
        """Runs Ray Tune. Uses fit_iter() internally for epoch reporting."""
```

The key difference from TSFast: no callback needed. `fit_iter()` naturally integrates with Ray Tune's reporting.

### 8. Visualization (`viz.py`)

Extend existing `plot_batch` / `plot_results` with optional type-specific rendering:

```python
def plot_batch(batch, plot_fn=plot_sequence):
    """plot_fn is swappable: plot_sequence (default), plot_spectrogram, plot_quaternion"""

def plot_results(pred, target, plot_fn=plot_sequence):
    """Same pattern."""
```

No type dispatch. Caller passes the appropriate plot function explicitly.

### 9. TBPTT (Truncated BPTT)

This is the hardest feature. Approach:

- `StatefulRNN` model variant that exposes `get_state()` / `set_state()` for hidden state
- `TBPTTSource` that yields ordered sub-sequences from the same file
- Grain's deterministic ordering guarantees sequence order within a batch
- State reset logic is **model-internal**: the model resets when it sees a new sequence ID

No callback needed. The model tracks state, the source provides ordered data.

---

## TSFast → tsjax Complete Feature Map

| TSFast Feature | tsjax Equivalent | Mechanism |
|---|---|---|
| `GradientClipping` | `optax.clip_by_global_norm()` | optax chain |
| `SkipNaNCallback` | `optax.zero_nans()` | optax chain |
| `GradientBatchFiltering` | `optax.masked()` | optax chain |
| `PhysicsLossCallback` | `compose_losses(base, physics, w)` | loss composition |
| `FranSysCallback` (5 terms) | `compose_losses(...)` | loss composition |
| `ConsistencyCallback` | loss composition | loss composition |
| `TransitionSmoothnessCallback` | loss composition | loss composition |
| `AuxiliaryOutputLoss` | `masked_loss(fn, mask)` | loss composition |
| `SkipNLoss` / `CutLoss` | `skip_n(fn, n)` | loss composition |
| `CB_AddLoss` | `compose_losses()` | loss composition |
| `PredictionCallback` | Model-internal (AR concatenation) | model-internal |
| `TbpttResetCB` | Model-internal (state management) | model-internal |
| `AlternatingEncoderCB` | Model-internal (jax.random switch) | model-internal |
| `ARInitCB` | Model-internal | model-internal |
| `SeqNoiseInjection` | `batch_transform=add_noise(...)` | batch_transform |
| `SeqBiasInjection` | `batch_transform=add_bias(...)` | batch_transform |
| `VarySeqLen` / `CB_TruncateSequence` | `batch_transform=truncate_schedule(...)` | batch_transform |
| `QuaternionAugmentation` | `batch_transform=quat_augment(...)` | batch_transform |
| `CBRayReporter` | `fit_iter()` + `ray.tune.report()` | fit_iter |
| Early stopping | `fit_iter()` + break | fit_iter |
| Checkpointing | `fit_iter()` + Orbax save | fit_iter |
| `WeightedDL_Factory` | `weighted_source()` | Grain sampling |
| `BatchLimit_Factory` / `NBatches_Factory` | `limit_batches()` | Grain sampling |
| `InferenceWrapper` | `predict()` (trivial — model is raw-in/raw-out) | inference |
| `HPOptimizer` | `HPOptimizer` using `fit_iter()` | tune |
| `Spectrogram` / `SpectrogramBlock` | `batch_transform` + JAX STFT | batch_transform |
| Scalar inputs | Multi-signal `SignalStore` | data layer |
| Type-dispatched visualization | Explicit `plot_fn` parameter | viz |

---

## Implementation Sequence

Ordered by dependency — each step is independently useful:

1. **`fit_iter()` + `batch_transform`** — foundation for everything else
2. **`losses/compose.py`** — compose_losses, skip_n, masked_loss
3. **`transforms/noise.py`** — noise/bias injection
4. **`transforms/truncate.py`** — sequence truncation schedules
5. **`data/sampling.py`** — weighted sampling, batch limiting
6. **`models/tcn.py`** — TCN architecture
7. **`models/crnn.py`** — CRNN (depends on TCN + RNN)
8. **`inference.py`** — predict utility
9. **`tune.py`** — Ray Tune HPOptimizer (depends on fit_iter)
10. **TBPTT** — StatefulRNN + TBPTTSource (complex, standalone)
11. **`models/ar.py`** — autoregressive / FranSys
12. **`models/pirnn.py` + `losses/physics.py`** — physics-informed (depends on compose, jax.grad)
13. **`transforms/quaternion.py` + `losses/quaternion.py`** — quaternion augmentation
14. **`viz.py` extensions** — spectrogram/quaternion plot functions

---

## Verification

After step 1 (fit_iter + batch_transform):
- Existing tests still pass: `UV_CACHE_DIR=/tmp/claude/uv-cache uv run pytest tests/ -v`
- `fit()` and `fit_flat_cos()` produce identical results (they now wrap fit_iter internally)
- New test: `fit_iter()` can be consumed by external loop and stopped early

After each subsequent step:
- Unit tests for the new module
- Integration test showing the feature works end-to-end with `Learner`
- Lint: `UV_CACHE_DIR=/tmp/claude/uv-cache uv run ruff check tsjax/ tests/`
