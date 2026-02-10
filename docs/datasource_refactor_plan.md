# Simplify data source architecture: fuse 5 classes into `DataSource`

## Context

The manual pipeline in `examples/00_minimal_example_jax.py` requires understanding 5 internal classes (`ComposedSource`, `_WindowIndex`, `_FileIndex`, `WindowedReader`, `FullSeqReader`) plus private imports. Train/valid/test always share the same specs (modalities) — only the windowing strategy varies. A single user-facing class can encapsulate the index + reader wiring while supporting all modalities and custom extensions.

**Before:** ~40 lines, 10 imports (2 private), 5 classes to understand
**After:** ~12 lines, 3 imports, 1 class to understand

---

## `DataSource` API

```python
class DataSource:
    """Grain-compatible time series source.

    win_sz=None → one full sequence per file.
    win_sz=int  → sliding windows with stp_sz stride.

    specs values can be:
      - list[str]        → auto-built windowed/full-seq reader
      - ScalarAttr(...)   → auto-built scalar attribute reader
      - Feature(...)      → auto-built feature reduction reader
      - any Reader object → used as-is (must be callable with .signals attr)
    """

    def __init__(
        self,
        store: SignalStore,
        specs: dict[str, ReaderSpec | Reader],
        *,
        win_sz: int | None = None,
        stp_sz: int = 1,
    ):
        self.specs = specs                 # stored for stats computation
        self.readers = _build_readers(     # dict[str, Reader]
            specs, store, list(store.paths),
            windowed=(win_sz is not None),
        )
        if win_sz is not None:
            ref = _find_ref_signal(specs)
            self._index = _WindowIndex(store, win_sz, stp_sz, ref)
        else:
            self._index = _FileIndex(list(store.paths))
        self._validate_lengths()

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> dict[str, np.ndarray]:
        path, l_slc, r_slc = self._index.resolve(idx)
        return {k: reader(path, l_slc, r_slc) for k, reader in self.readers.items()}
```

**Key properties:**
- `.readers[key].signals` preserved → plot labels in `Learner` still work
- `.specs` stored → `GrainPipeline.from_sources()` can auto-compute stats
- Custom reader objects pass through `_build_readers` unchanged → extensibility without modifying tsjax

**Reader protocol (informal):**
```python
class Reader(Protocol):
    signals: list[str]
    def __call__(self, path: str, l_slc: int, r_slc: int) -> np.ndarray: ...
```

---

## `GrainPipeline.from_sources()` API

```python
@classmethod
def from_sources(
    cls,
    train: DataSource,
    valid: DataSource,
    test: DataSource,
    *,
    input_keys: tuple[str, ...],
    target_keys: tuple[str, ...],
    bs: int = 64,
    seed: int = 42,
    stats: dict[str, NormStats] | None = None,  # override auto-computed stats
) -> GrainPipeline:
```

When `stats=None`, auto-computes from `train.specs` + `train.readers`:
- `list[str]` → `compute_norm_stats_from_index(reader.store, reader.signals)`
- `ScalarAttr` → `compute_scalar_stats(index.file_paths, spec.attrs)`
- `Feature` → `compute_stats_with_transform(train, key, spec.fn)`
- Custom reader → requires explicit `stats` (raises ValueError if missing)

---

## Example (after)

```python
# %% Simple factory
pipeline = create_simulation_dls(
    u=['u'], y=['y'],
    dataset=_root / 'test_data/WienerHammerstein',
    bs=16, win_sz=500, stp_sz=10, preload=True,
)
lrn = RNNLearner(pipeline, rnn_type='lstm', hidden_size=64, n_skip=10, metrics=[rmse])
lrn.fit_flat_cos(n_epoch=1, lr=1e-3)

# %% Manual pipeline with custom stores
DATASET = _root / "test_data/WienerHammerstein"
split_files = {
    s: sorted(str(p) for p in (DATASET / s).rglob("*.hdf5"))
    for s in ("train", "valid", "test")
}
stores = {s: HDF5Store(files, ["u", "y"], preload=True) for s, files in split_files.items()}
specs = {"u": ["u"], "y": ["y"]}

train_src = DataSource(stores["train"], specs, win_sz=500, stp_sz=10)
valid_src = DataSource(stores["valid"], specs, win_sz=500, stp_sz=10)
test_src  = DataSource(stores["test"], specs)

pipeline2 = GrainPipeline.from_sources(
    train_src, valid_src, test_src,
    input_keys=("u",), target_keys=("y",), bs=16,
)
lrn2 = RNNLearner(pipeline2, rnn_type="lstm", hidden_size=64, n_skip=10)
lrn2.fit(n_epoch=1, lr=1e-3)
```

---

## Changes by file

### `tsjax/data/sources.py`
1. Add `DataSource` class (as designed above)
2. Move `_build_readers` from `pipeline.py` here — extend it to pass through callable reader objects:
   ```python
   if callable(spec):
       readers[key] = spec  # pre-built reader, use as-is
   elif isinstance(spec, list): ...
   ```
3. Add `_find_ref_signal(specs)` helper (extracted from `create_grain_dls`)
4. Rename `WindowedReader` → `_WindowedReader`
5. Keep `_WindowIndex`, `_FileIndex`, `FullSeqReader`, `ScalarAttrReader`, `FeatureReader` as internal
6. Keep `ComposedSource` temporarily as `_ComposedSource` alias (only if tests need it, otherwise remove)

### `tsjax/data/pipeline.py`
1. Simplify `create_grain_dls` source construction (lines 250-280 → ~6 lines):
   ```python
   train_source = DataSource(train_store, all_specs, win_sz=win_sz, stp_sz=stp_sz)
   valid_source = DataSource(valid_store, all_specs, win_sz=win_sz, stp_sz=valid_stp_sz)
   test_source  = DataSource(test_store, all_specs)
   ```
   Pure-scalar case (`store=None`): handle with `_FileIndex` + `ScalarAttrReader` inline (rare edge case)
2. Add `GrainPipeline.from_sources()` classmethod
3. Add `_compute_stats_from_source(source)` helper
4. Remove `_build_readers` (moved to sources.py)
5. Update `GrainPipeline` type annotations: `ComposedSource` → `DataSource`

### `tsjax/data/__init__.py`
- Add: `DataSource`
- Remove: `ComposedSource`, `WindowedReader`, `FullSeqReader`
- Keep: `ScalarAttrReader`, `FeatureReader` (needed if users build custom readers that delegate)

### `tsjax/__init__.py`
- Replace `ComposedSource` → `DataSource`

### `examples/00_minimal_example_jax.py`
- Rewrite manual path section as shown above

### `tsjax/training/learner.py`, `tsjax/training/factory.py`
- Update type annotations only (if any reference `ComposedSource`)
- No logic changes — `.readers[key].signals` access preserved

### Tests
- `tests/test_data_pipeline.py` — update source construction to use `DataSource`
- `tests/test_resample.py` — update imports (`_WindowedReader`, `DataSource`)
- Other test files using `create_grain_dls` — no changes needed

---

## Extensibility summary

| Extension | What the user does | tsjax changes needed |
|---|---|---|
| New store format | Implement `SignalStore` protocol, pass to `DataSource` | None |
| Custom reader logic | Pass callable reader object in `specs` dict | None |
| New built-in ReaderSpec | Add dataclass + reader class + `_build_readers` branch + stats branch | ~24 lines across 3 files |

---

## Verification
1. `UV_CACHE_DIR=/tmp/claude/uv-cache uv run ruff check tsjax/ tests/ examples/`
2. `UV_CACHE_DIR=/tmp/claude/uv-cache uv run pytest tests/ -v`
3. Verify `source.readers[key].signals` works (learner/factory tests)
4. Verify example 00 manual path uses only public API
