# Lessons from Real-World TsFast Usage: Analysis of Hendrik's Virtual Wheel Speed Sensor Project

> **Source project:** `/Users/daniel/Development/code_hendrik/`
> **Author of source project:** Hendrik Schaefke
> **Analysis date:** 2025-02-11
> **Purpose:** Inform the JAX rewrite of TsFast (tsjax) by critically reviewing what a real research user had to build on top of TsFast.

---

## 1. Project Context

Hendrik built a **virtual wheel speed sensor** using RNN/TCN architectures with Ray Tune HPO (hyperparameter optimization). The project trains neural networks to predict wheel speed signals from vehicle CAN bus data (engine speed, wheel velocities, torque, acceleration) across multiple vehicles (ID4, ID7, IDBuzz) and simulation data (CarMaker).

### Project structure (flat, no subdirectories)

| File | Size | Purpose |
|---|---|---|
| `helper.py` | 3,415 lines | Custom utilities: scalers, data loaders, callbacks, transforms, plotting, export |
| `wheel_speed_sensor.py` | 533 lines | CLI training script with Ray Tune HPO |
| `wheel_speed_sensor_hybrid.py` | 450 lines | Alternative HPO script with multi-dataset support |
| `wheel_speed_sensor.ipynb` | 7.0 MB | Main interactive notebook (125 cells) |
| `plots.ipynb` | 3.4 MB | HPO result visualization |
| `13_Callbacks.ipynb` | 3.9 MB | Callback experimentation |
| `12_Data_Augmentation.ipynb` | 360 KB | Augmentation development |
| `11_MinMax_Scaling.ipynb` | 303 KB | Custom scaler development |

The critical observation: **`helper.py` alone is 3,415 lines** -- roughly the complexity of a small library. This is the clearest signal that TsFast forced Hendrik to build a parallel infrastructure just to do his research.

### Tech stack

- **Core:** PyTorch, TsFast, fastai
- **HPO:** Ray Tune with ASHA scheduler
- **Data:** HDF5 files via `asammdf`, `h5py`
- **Models:** GRU, LSTM, RNN, TCN (via TsFast's `RNNLearner`, `TCNLearner`)
- **Deployment target:** JSON export for embedded systems / MATLAB

---

## 2. TsFast API Surface Used

### Imports

```python
# helper.py
from tsfast.basics import *
from tsfast.data import *
from tsfast.data.core import TensorSequencesInput, TensorSequencesOutput
from tsfast.tune import CBRayReporter

# wheel_speed_sensor.py / wheel_speed_sensor_hybrid.py
from tsfast.basics import *
from tsfast.tune import *
```

### Classes and functions consumed from TsFast

| API | Usage |
|---|---|
| `RNNLearner(dls, rnn_type, n_skip, num_layers, hidden_size, loss_func, metrics, opt_func, input_p, hidden_p, weight_p)` | Create RNN/LSTM/GRU learner |
| `TCNLearner(dls, num_layers, hidden_size, loss_func, metrics, opt_func)` | Create TCN learner |
| `HPOptimizer(create_lrn_fn, dls)` | Ray Tune integration |
| `HPOptimizer.start_ray()` / `.optimize(config, ...)` | Launch HPO |
| `SequenceBlock.from_hdf(clm_names, seq_cls, cached)` | Data block for HDF5 sequences |
| `TensorSequencesInput` / `TensorSequencesOutput` | Type markers for transform dispatch |
| `CBRayReporter` | Built-in Ray Tune callback |
| `HDF2Sequence` | HDF5 to tensor conversion |
| `get_hdf_files(path, folders)` | HDF5 file discovery |
| `ParentSplitter()` | Split by parent folder name |
| `CreateDict`, `DfApplyFuncSplit`, `DfHDFCreateWindows` | Data pipeline internals |
| `BatchLimit_Factory(TfmdDL)` / `BatchLimit_Factory(TbpttDl)` | DataLoader with batch limits |
| `Normalize(mean, std, axes)` | Batch normalization transform |
| `InferenceWrapper(learner)` | Inference-time model wrapper |
| `fun_rmse` | Built-in RMSE metric |
| `extract_mean_std_from_dataset` | Dataset statistics extraction |
| `dict_file_load` / `dict_file_save` | Normalization cache |
| `save_model` | Model serialization |
| `trial_dir_creator` | HPO trial naming |

### DataLoader attributes accessed

- `dls.clm_names[0]` / `dls.clm_names[1]` -- input/output channel names
- `dls[-1]` -- test dataloader
- `dls.decode((None, tensor))` -- decode normalized predictions
- `dls[-1].items` -- file paths in test set
- `dls.after_batch` -- batch transforms (to find scalers)
- `dls.train` / `dls.valid` -- split dataloaders
- `dls.loaders.append(test_dl)` -- manually adding test dataloader

---

## 3. Critical Review: What Hendrik Had to Build Himself

### 3.1 Scaling / Normalization (~350 lines)

**What he built:** Three custom scaler classes (`MinMaxScaler`, `MaxAbsScaler`, `StandardScaler`), a `SequenceBlockRaw` to disable TsFast's hardcoded normalization, and five helper functions to extract statistics from HDF5 files.

**The root problem:** TsFast's `SequenceBlock` hardcodes `Normalize` as a batch transform. There is no way to opt out or choose a different strategy without subclassing the entire block:

```python
# Hendrik's workaround (helper.py:1205-1220): clone SequenceBlock, disable normalization
class SequenceBlockRaw(TransformBlock):
    """A tsfast SequenceBlock WITHOUT automatic normalization."""
    def __init__(self, seq_extract, padding=False):
        return super().__init__(type_tfms=[seq_extract],
                                # batch_tfms=[Normalize(axes=[0,1])],  # DISABLED
                                dls_kwargs={})
```

**Why this is bad design in TsFast:**

- Normalization strategy is a **user decision**, not a library default
- TsFast only normalizes **inputs**, never outputs -- but output scaling is critical for multi-scale regression (e.g., when wheel velocity in m/s and torque in Nm are both targets)
- The `Normalize` transform uses mean/std only; min-max, max-abs, and no-scaling are equally valid
- Computing scaling statistics requires scanning the full dataset, but TsFast doesn't expose this machinery for alternative strategies

**The cascade effect:** Because he couldn't control scaling, Hendrik also needed:

- `extract_min_max_from_hdffiles()` (helper.py:1255-1287) -- manual min/max stats computation
- `extract_min_max_from_dataset()` (helper.py:1290-1311) -- wrapper using only training files
- `make_fun_rmse_custom()` (helper.py:2208-2214) -- a closure to evaluate RMSE on **decoded** (original-scale) predictions, since metrics are computed on normalized data but the domain requires m/s
- 4 variants of `create_dls_*` functions (helper.py:1499-2170), each with different scaling plumbing

**Custom scaler implementations** (all follow the same pattern):

```python
# helper.py:1314-1373 -- MinMaxScaler
class MinMaxScaler(DisplayedTransform):
    def __init__(self, min_in, max_in, min_out=None, max_out=None):
        store_attr()

    def encodes(self, x: TensorSequencesInput):
        if x.device != self.min_in.device:  # hidden device migration
            self.min_in, self.max_in = self.min_in.to(x.device), self.max_in.to(x.device)
        return (x - self.min_in) / (self.max_in - self.min_in)

    def decodes(self, x: TensorSequencesInput):  # inverse
        return x * (self.max_in - self.min_in) + self.min_in

    def encodes(self, x: TensorSequencesOutput):  # optional output scaling
        if self.min_out is None: return x
        return (x - self.min_out) / (self.max_out - self.min_out)

# helper.py:1436-1497 -- StandardScaler (same pattern with mean/std)
# helper.py:1377-1433 -- MaxAbsScaler (same pattern with maxabs)
```

All three share identical boilerplate: `store_attr()`, device migration in `encodes`, separate `encodes` for input/output types, inverse in `decodes`.

### 3.2 DataLoader Creation (~670 lines, 4 variants)

**What he built:**

| Function | Lines | Key Difference from TsFast's `create_dls` |
|---|---|---|
| `create_dls_custom` (1499-1608) | 110 | Adds `item_tfms`, `batch_tfms` params; no prediction/input_delay |
| `create_dls_custom_param` (1610-1767) | 160 | + user-provided `mean`/`std` (scalar or per-channel) |
| `create_dls_from_filelists` (1787-1953) | 170 | + explicit train/valid/test/scale file lists; dataset subsampling |
| `create_dls_custom_scale` (1956-2170) | 215 | + scaler strategy selector (`none`/`normalize`/`standard`/`minmax`/`maxabs`) |

**The root problem:** TsFast's `create_dls()` is a monolithic function that couples:

1. File discovery (assumes `train/valid/test` folder structure under one root)
2. Windowing (one config for all splits, except valid stride)
3. Normalization (hardcoded mean/std `Normalize`)
4. DataLoader construction (batch sizes, workers, TBPTT)
5. Test DataLoader creation

When any single piece doesn't fit the user's needs, the entire function must be rewritten. Hendrik's code shows the progressive escalation:

1. Started with `create_dls_custom` -- needed item transforms
2. Then `create_dls_custom_param` -- needed to control normalization values
3. Then `create_dls_from_filelists` -- needed to compose datasets from multiple sources
4. Finally `create_dls_custom_scale` -- needed to choose between 5 scaling strategies

**Specific pain points visible in code:**

- `helper.py:1535-1542` -- caching normalization stats with `dict_file_load/save` because recomputing from HDF5 is slow for large datasets
- `helper.py:1929-1943` -- monkey-patching `train_dl.items` and `train_dl.n` for dataset subsampling (fraction of training data for ablation studies)
- `wheel_speed_sensor.py:14` -- TODO: "Dataloader soll auch Liste von Pfaden zu Datensaetzen bekommen koennen" (DataLoader should accept a list of dataset paths)
- `select_files_by_source()` (helper.py:2332-2389, 60 lines) exists solely to mix experimental and simulated data per split

**Code duplication:** All 4 `create_dls_*` functions share ~80% identical boilerplate (file collection, DataBlock construction, TBPTT branching, test DL creation). The differences are only in scaling setup.

### 3.3 Data Augmentation (~115 lines)

**What he built:** `SymmetryAugmentItem` (helper.py:2216-2330) -- a physics-aware augmentation that swaps left/right wheel channels and negates lateral signals, exploiting vehicle dynamics symmetry.

```python
# Usage (wheel_speed_sensor.py:276-281):
item_tfms_symmetry = [SymmetryAugmentItem(
    swap_in=[("wheel_vel_rl_CAN_ms","wheel_vel_rr_CAN_ms"),
             ("wheel_vel_fl_CAN_ms","wheel_vel_fr_CAN_ms")],
    swap_out=[("wheel_vel_rl_WSS_ms", "wheel_vel_rr_WSS_ms")],
    p=0.3
)]
```

**What works well about this design:**

- References channels by **name** (not index), resolving via `dl.clm_names` in `setups()`
- Warns about unknown channels gracefully instead of crashing
- Applies only during training (`split_idx = 0`)
- Independent probability per item
- Handles both input and output channels via type dispatch

**What TsFast should learn:**

- Transforms need to know **channel semantics**. The current type dispatch (`TensorSequencesInput` vs `TensorSequencesOutput`) is necessary but insufficient -- transforms also need channel names.
- The `setups(dl)` hook that reads `dl.clm_names` is the right pattern but shouldn't require manual implementation every time.
- Domain-specific augmentations (symmetry, physics constraints) are a real need; the library should provide the **framework** (name resolution, train-only application, paired input/output transforms) even if specific augmentations are user-defined.

### 3.4 Ray Tune Callbacks (~200 lines, 3+1 variants)

**What he built:**

| Callback | Lines | Purpose |
|---|---|---|
| `CBRayReporter_custom` (998-1016) | 19 | Reports metrics + checkpoint every epoch |
| `CBRayReporter_custom_end` (1018-1068) | 51 | Reports metrics every epoch, saves ONE final checkpoint |
| `CBRayReporterCkptCtrl` (1072-1130) | 59 | Configurable: checkpoint every N epochs + optional end checkpoint |
| `PredictionLoggerCallback` (1132-1202) | 71 | Logs per-epoch predictions to CSV using InferenceWrapper |

**The root problem with `CBRayReporter`:** TsFast's built-in `CBRayReporter` saves a checkpoint **every epoch**. With 200 HPO samples x 300 epochs, this creates 60,000 checkpoints. There's no way to configure frequency or disable intermediate checkpoints. Hendrik wrote 3 variants trying to find the right balance:

1. First attempt: `CBRayReporter_custom` -- full copy with minor tweaks
2. Second: `CBRayReporter_custom_end` -- report-only during training, one final checkpoint
3. Third: `CBRayReporterCkptCtrl` -- finally found the right abstraction (configurable frequency + end flag)

All three share identical metric-gathering boilerplate (`_gather_metrics` extracting from `self.learn.recorder.values[-1]`).

**The `PredictionLoggerCallback` reveals a deeper problem:** Getting predictions during training requires:

1. Creating an `InferenceWrapper` in `before_fit()`
2. Manually syncing model weights each epoch
3. Checking `hasattr(self, "gather_preds")` to distinguish training from evaluation
4. Building a DataFrame with epoch-indexed columns (`ep0_y1`, `ep1_y1`, ...)

This should be a one-liner, not 71 lines.

### 3.5 Model Export (~160 lines)

**What he built:** `export_GRU()` (helper.py:3089-3245+) -- serializes a trained GRU to JSON with:

- Meta: input/output sizes, column names, normalization parameters
- Weights: all GRU gate weights (reset, update, new) decomposed per layer
- Output head: linear layer weights and bias

**Why this exists:** The model needs to run in **embedded systems / MATLAB / C++**. Standard `torch.save()` doesn't include:

- Normalization parameters (mean, std, min, max)
- Channel names and their mapping
- Decomposed gate weights (reset/update/new gates separately, not as a combined tensor)
- A format readable outside Python

This is a **deployment concern that TsFast ignores entirely**. For automotive engineering, training is only half the job -- the model must be deployable on an ECU.

### 3.6 Visualization (~900 lines)

Nearly a third of `helper.py` is plotting code. Key functions:

| Function | Purpose |
|---|---|
| `plot_loss()` / `plot_loss_hpo()` | Loss curves (matplotlib) |
| `plot_loss_hpo_plotly()` | Interactive loss curves for 200+ trials |
| `plot_parallel_coordinates_plotly()` | Hyperparameter importance visualization |
| `plot_preds_plotly()` | Predictions vs targets (interactive) |
| `plot_grid_boxplots()` / `plot_grid_meanswarm()` | Metric comparison grids |
| `plot_grid_heatmap()` / `plot_dendro_heatmap()` | Heatmap visualizations |
| `build_epoch_df()` | Epoch-level DataFrame construction |
| `plot_train_valid_by_config()` | Train/validation comparison by config |

These are all **standard needs** for anyone doing HPO with time series models. TsFast provides none of them.

### 3.7 Data Management Utilities (~200 lines)

| Function | Purpose |
|---|---|
| `select_files_by_source()` (2332-2389) | Mix sim/exp data per split |
| `collect_hdf_files()` (1769-1785) | Multi-dataset file collection |
| `analyze_dataset_duration()` (2414-2476) | Signal value distribution analysis |
| `print_dls_info()` (2172-2206) | DataLoader debug info |
| `save_predictions_to_hdf5()` (2479-2530) | Write predictions back to source HDF5 |
| `save_predictions_as_hdf5()` (2532-2590) | Save predictions as new HDF5 file |
| `copy_best_model_pth()` (2593-2641) | Copy best HPO checkpoint to results dir |
| `my_setup()` (2726-2829) | Print environment info (versions, GPU, etc.) |

### 3.8 Custom Metrics (~80 lines)

```python
# helper.py:2208-2214 -- decoded RMSE (closure over dls)
def make_fun_rmse_custom(dls):
    def fun_rmse_decoded(inp, targ):
        _, targ_dec = dls.decode((None, targ))
        _, pred_dec = dls.decode((None, inp))
        return torch.sqrt(F.mse_loss(pred_dec, targ_dec))
    return fun_rmse_decoded

# Also: error_mae, error_mse, error_mbe, error_shape_osc (domain-specific)
```

The closure pattern for decoded metrics is a **workaround** for TsFast not supporting original-scale evaluation natively.

---

## 4. What Should Be in TsFast vs. What Is Application-Specific

### Belongs in TsFast (library-level concerns)

| Feature | Lines Hendrik wrote | Justification |
|---|---|---|
| Pluggable scaling strategies | ~180 (3 scalers) | Every user needs this; hardcoded Normalize is the #1 pain point |
| Output scaling | ~90 (output `encodes`/`decodes` in each scaler) | Multi-scale regression is standard |
| SequenceBlock without auto-normalization | ~20 (SequenceBlockRaw) | Should be a parameter, not a subclass |
| Flexible data source composition | ~60 (select_files_by_source) | Mixing datasets per split is universal in research |
| Dataset statistics extraction | ~100 (extract_min_max_* helpers) | Infrastructure for any non-mean/std scaler |
| Statistics caching | ~20 (dict_file_load/save calls) | Scanning large HDF5 datasets is slow |
| Configurable checkpoint callbacks | ~130 (3 callback variants) | Checkpoint-every-epoch is wasteful for large HPO |
| Decoded metrics | ~10 (make_fun_rmse_custom) | Evaluating on original scale is universal |
| DataLoader from explicit file lists | ~170 (create_dls_from_filelists) | Multi-source, multi-vehicle experiments are common |
| Named channel transforms | ~50 (name resolution in SymmetryAugmentItem) | Transforms should reference channels by name |
| Model export with metadata | ~160 (export_GRU) | JSON/ONNX export is a deployment basic |
| HPO result visualization | ~400 (plot functions) | Loss curves, parallel coordinates, prediction plots |
| DataLoader info printing | ~35 (print_dls_info) | Standard debugging need |

**Total: ~1,425 lines that should have been library features.**

### Application-specific (should NOT be in TsFast)

| Feature | Lines | Reason |
|---|---|---|
| Vehicle symmetry augmentation specifics | ~65 | Domain-specific; but the framework for channel-aware augmentation should exist |
| Input variable set mappings (`u_vars_1..5`) | ~20 | Project configuration |
| MDF/ASAM file export | ~200 | Automotive-specific format |
| Dataset duration analysis | ~65 | Domain-specific analytics |
| Corporate color schemes and plot styling | ~100 | Project cosmetics |
| HPO config per model type | ~95 | Project-specific search spaces |
| Prediction-to-HDF5 export | ~110 | Automotive workflow-specific |

**Total: ~655 lines that are genuinely application-specific.**

### Verdict

Of Hendrik's 3,415 lines, roughly **1,425 (42%) should have been TsFast features**, **655 (19%) are genuinely application-specific**, and the remaining **1,335 (39%) are visualization/analysis boilerplate** that a library should at least partially address.

---

## 5. Architectural Recommendations for tsjax

### 5.1 Core Principle: Composable Pipelines, Not Monolithic Functions

The current TsFast forces users through `create_dls()` which bundles 6 concerns. The JAX rewrite should use **independent, composable stages**:

```python
# CURRENT TsFast (monolithic -- forces rewrite when any piece doesn't fit)
dls = create_dls(dataset_path, u, y, win_sz=200, stp_sz=1, bs=64)

# PROPOSED tsjax (composable -- replace any stage independently)
dataset = tsjax.Dataset.from_hdf(
    sources={"ID7": path_id7, "SIM": path_sim},
    inputs=["wheel_vel_rl", "wheel_vel_rr", "engine_vel"],
    outputs=["wheel_vel_rl_WSS", "wheel_vel_rr_WSS"],
)

dataset = dataset.window(size=200, stride=1)
dataset = dataset.split(
    train={"ID7": "train", "SIM": "train"},
    valid={"ID7": "valid"},
    test={"ID7": "test", "ID4": "test", "IDBuzz": "test"},
)

scaler = tsjax.scalers.Standard(fit_on="train")  # or MinMax, MaxAbs, Identity
dataset = dataset.scale(inputs=scaler, outputs=scaler)  # explicit control

augment = tsjax.augment.ChannelSymmetry(
    swap_inputs=[("wheel_vel_rl", "wheel_vel_rr")],
    swap_outputs=[("wheel_vel_rl_WSS", "wheel_vel_rr_WSS")],
    p=0.3,
)
dataset = dataset.augment(augment, split="train")

train_dl, valid_dl, test_dl = dataset.to_loaders(
    batch_size={"train": 64, "valid": 16, "test": 1},
    max_batches={"train": 1000},
)
```

Each step is independent, testable, and replaceable. A user who needs custom scaling just swaps the scaler; a user who needs custom splits just changes the split config.

### 5.2 Scaling System: First-Class, Pluggable, Bidirectional

```python
# Built-in scalers
scaler = tsjax.scalers.Standard()    # z-score: (x - mean) / std
scaler = tsjax.scalers.MinMax()      # [0, 1] range
scaler = tsjax.scalers.MaxAbs()      # [-1, 1] by max absolute value
scaler = tsjax.scalers.Identity()    # explicit no-op

# Key capabilities:
# 1. Fit on specific data subset (training set only)
scaler.fit(train_data)

# 2. Apply to inputs, outputs, or both -- INDEPENDENTLY
dataset.scale(inputs=scaler_in, outputs=scaler_out)

# 3. Inverse transform predictions back to original scale
original_preds = scaler_out.inverse(normalized_preds)

# 4. Access statistics
scaler.mean_, scaler.std_, scaler.min_, scaler.max_

# 5. Serialize alongside model for deployment
tsjax.export(model, scaler=scaler, path="model.json")

# 6. Custom scalers via simple protocol
class MyScaler:
    def fit(self, data): ...
    def transform(self, data): ...
    def inverse(self, data): ...
```

This eliminates: all 3 custom scaler classes, `SequenceBlockRaw`, all `extract_*` helpers, and `make_fun_rmse_custom`.

### 5.3 Metrics: Always Offer Original-Scale Evaluation

```python
# Metrics should automatically decode if a scaler is present
model = tsjax.RNN(dls, metrics=[tsjax.metrics.RMSE(decoded=True)])

# Under the hood: metric accesses the scaler from the data pipeline
# No closure hacks like make_fun_rmse_custom(dls) needed
```

### 5.4 Multi-Source Dataset Composition

```python
# Declare sources
sources = tsjax.Sources({
    "ID7": "/data/Vehicle_ID7_50Hz",
    "ID4": "/data/Vehicle_ID4",
    "IDBuzz": "/data/Vehicle_IDBuzz",
    "SIM": "/data/CarMaker_small",
})

# Compose splits flexibly
splits = tsjax.Splits(
    train=sources.select("ID7", split="train"),
    valid=sources.select("ID7", split="valid"),
    test=sources.select(["ID7", "ID4", "IDBuzz"], split="test"),
    scale=sources.select("ID7", split="train"),  # fit scaler on this subset
)
```

This replaces: `select_files_by_source()`, `collect_hdf_files()`, `create_dls_from_filelists()`, and the multiple dataset path constants scattered across scripts.

### 5.5 Channel-Aware Transform System

```python
# Transforms reference channels by NAME, resolved automatically
augment = tsjax.augment.ChannelSymmetry(
    swap_inputs=[("wheel_vel_rl", "wheel_vel_rr")],
    negate_inputs=["acc_y", "yaw_rate"],
    swap_outputs=[("wheel_vel_rl_WSS", "wheel_vel_rr_WSS")],
    p=0.3,
)

# The library resolves names -> indices from the dataset metadata
# Warns about unknown channels (like Hendrik's impl does)
# Applies only during training by default
```

### 5.6 Callback System: Configurable, Not Hardcoded

```python
# HPO reporter with configurable checkpointing
reporter = tsjax.callbacks.HPOReporter(
    checkpoint_frequency=10,      # every 10 epochs (0 = never mid-training)
    checkpoint_at_end=True,       # always save final checkpoint
    log_predictions=True,         # optional per-epoch prediction logging
    prediction_sample_idx=0,      # which sample to track
)

# Built-in prediction tracking (replaces PredictionLoggerCallback)
tracker = tsjax.callbacks.PredictionTracker(
    dl_idx=-1, sample_idx=0,
    save_path="predictions.csv",
)
```

### 5.7 Model Export: First-Class Deployment Support

```python
# Export model with full metadata
tsjax.export.to_json(
    model,
    path="model.json",
    scaler=scaler,              # includes normalization params
    channel_names=dataset.channel_names,  # input/output names
    decompose_gates=True,       # GRU/LSTM gate weights separately
)

# Also support standard formats
tsjax.export.to_onnx(model, path="model.onnx")
tsjax.export.to_jax(model, path="model_weights.npz")  # pure JAX arrays
```

### 5.8 HPO Integration: Opinionated but Overridable

```python
# High-level: one-liner HPO (what most users want)
results = tsjax.hpo.optimize(
    model_fn=create_model,
    dataset=dataset,
    config=tsjax.hpo.Config(
        hidden_size=tsjax.hpo.choice([16, 32, 64]),
        lr=tsjax.hpo.loguniform(1e-4, 1e-2),
        scaler=tsjax.hpo.choice(["standard", "minmax"]),
    ),
    scheduler=tsjax.hpo.ASHA(grace_period=50),
    num_samples=200,
    checkpoint_frequency=0,  # only at end
)

# Low-level: full control (Ray Tune directly with tsjax components)
```

### 5.9 Built-In Visualization

```python
# HPO results
tsjax.plot.hpo_loss_curves(results, top_n=10)
tsjax.plot.parallel_coordinates(results, metric="rmse")
tsjax.plot.hpo_importance(results)

# Predictions
tsjax.plot.predictions(model, test_dl, sample_idx=0, fs=50)

# All plots: matplotlib by default, plotly=True for interactive
```

---

## 6. JAX-Specific Design Considerations

### 6.1 Functional Transforms (not class-based with hidden state)

JAX is functional; the transform system should match:

```python
# Instead of fastai's DisplayedTransform with mutable encodes/decodes:
scaler_params = tsjax.scalers.standard_fit(train_data)
normalized = tsjax.scalers.standard_apply(data, scaler_params)
original = tsjax.scalers.standard_inverse(normalized, scaler_params)

# Scaler params are just a pytree (NamedTuple/dataclass):
# StandardParams(mean=jnp.array(...), std=jnp.array(...))
# Easily serializable, no hidden state, no device migration
```

### 6.2 JIT-Compatible Data Pipeline

```python
# Scaling and augmentation should be JIT-compilable
@jax.jit
def preprocess_batch(batch, scaler_params, augment_key):
    x, y = batch
    x = tsjax.scalers.standard_apply(x, scaler_params.inputs)
    y = tsjax.scalers.standard_apply(y, scaler_params.outputs)
    x, y = tsjax.augment.channel_symmetry(x, y, config, augment_key)
    return x, y
```

### 6.3 No Hidden Mutable State

The biggest problem with fastai's transform system is hidden mutable state (device migration in `encodes`, lazy initialization in `setups`). In JAX:

```python
# BAD (current TsFast pattern - mutable state):
class MinMaxScaler(DisplayedTransform):
    def encodes(self, x):
        if x.device != self.min_in.device:  # hidden state mutation!
            self.min_in = self.min_in.to(x.device)
        return (x - self.min_in) / (self.max_in - self.min_in)

# GOOD (JAX pattern - explicit params, no mutation):
def minmax_scale(x, params):
    return (x - params.min) / (params.max - params.min)
# params is a frozen pytree, passed explicitly, no device migration needed
```

### 6.4 Transparent Model Internals

```python
# Models as pure functions with explicit state
model = tsjax.models.GRU(hidden_size=64, num_layers=2)
params = model.init(rng, sample_input)

# Easy weight inspection (no .rnn.rnns[0].weight_ih_l0 digging)
jax.tree.map(lambda x: x.shape, params)

# Easy export: params is just a dict/pytree, directly serializable
```

---

## 7. Summary: The 5 Biggest Lessons

### Lesson 1: Don't hardcode normalization

It's the single decision that cascaded into the most custom code: 3 scaler classes, `SequenceBlockRaw`, stat extraction helpers, decoded metrics, 4 DataLoader variants. **Make scaling pluggable and bidirectional (inputs AND outputs) from day one.**

### Lesson 2: Separate data composition from data loading

Users need to mix sources, select per-split, subsample, and compose datasets flexibly. A monolithic `create_dls()` cannot serve this. **Use composable pipeline stages where each concern is independently configurable.**

### Lesson 3: Make channel names first-class

Transforms, augmentations, metrics, and exports all need to know what each feature dimension means. **Carry `channel_names` through the entire pipeline**, not just as an attribute on the DataLoader.

### Lesson 4: Design for deployment, not just training

Model export with normalization metadata, channel names, and decomposed weights is not optional -- it's why these models are being trained. **Build export into the core.**

### Lesson 5: Provide configurable defaults, not hardcoded behaviors

Checkpoint every epoch? Normalize inputs only? Use mean/std only? These are all reasonable defaults but must be overridable without subclassing or monkey-patching. **Use configuration objects with sensible defaults** rather than baking decisions into function internals.

### The north star metric

Hendrik's 3,415-line `helper.py` should shrink to **under ~200 lines** of truly application-specific code (vehicle symmetry definitions, MDF export, corporate plot styling). Everything else should be expressible through tsjax's API.

---

## Appendix: File-by-File Line Count of Custom Code

```
helper.py breakdown (3,415 lines):
  Plotting & visualization:     ~900 lines  (26%)
  DataLoader creation (4 variants): ~670 lines  (20%)
  Scaling system:               ~350 lines  (10%)
  Callbacks (4 variants):      ~200 lines   (6%)
  Data augmentation:            ~115 lines   (3%)
  Model export:                 ~160 lines   (5%)
  Data management utilities:    ~200 lines   (6%)
  Analysis helpers:             ~300 lines   (9%)
  MDF/HDF5 export:              ~200 lines   (6%)
  Misc (setup, paths, etc.):   ~320 lines   (9%)

wheel_speed_sensor.py (533 lines):
  Config & variable definitions: ~260 lines
  Learner creation:              ~115 lines
  HPO orchestration:             ~60 lines
  Post-processing:               ~98 lines

wheel_speed_sensor_hybrid.py (450 lines):
  Config & variable definitions: ~130 lines
  Learner creation:              ~80 lines
  HPO orchestration:             ~60 lines
  Post-processing:               ~180 lines
```
