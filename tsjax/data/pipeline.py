"""Factory function to create Grain data pipelines yielding raw data."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import grain

from .hdf5_store import HDF5Store
from .item_transforms import (
    Augmentation,
    Transform,
    _apply_augmentations,
    _apply_transforms,
    _MapOp,
    _RandomMapOp,
)
from .resample import ResampledStore, ResampleFn, resample_interp
from .sources import (
    DataSource,
    Feature,
    FeatureReader,
    ReaderSpec,
    ScalarAttr,
    ScalarAttrReader,
    SequenceReader,
)
from .stats import (
    NormStats,
    compute_stats,
)


def _make_sequential_loader(
    source: DataSource,
    operations: list,
    worker_count: int = 0,
) -> grain.DataLoader:
    """Create a deterministic DataLoader with SequentialSampler."""
    sampler = grain.samplers.SequentialSampler(
        num_records=len(source),
        shard_options=grain.sharding.NoSharding(),
    )
    return grain.DataLoader(
        data_source=source,
        sampler=sampler,
        operations=list(operations),
        worker_count=worker_count,
    )


@dataclass
class GrainPipeline:
    """Container for train/valid/test Grain DataLoaders with norm stats."""

    valid: grain.DataLoader
    test: grain.DataLoader
    stats: dict[str, NormStats]
    input_keys: tuple[str, ...]
    target_keys: tuple[str, ...]
    train_source: DataSource
    valid_source: DataSource
    test_source: DataSource
    n_train_batches: int
    _train_ops: list = field(repr=False)
    _train_worker_count: int = field(default=0, repr=False)
    _seed: int = field(default=42, repr=False)

    def train_loader(self, epoch: int = 0) -> grain.DataLoader:
        """Create a shuffled training DataLoader for the given epoch.

        Each epoch gets a different shuffle (seed + epoch).  The loader
        produces exactly one epoch of batches, then stops.
        """
        sampler = grain.samplers.IndexSampler(
            num_records=len(self.train_source),
            shuffle=True,
            seed=self._seed + epoch,
            num_epochs=1,
            shard_options=grain.sharding.NoSharding(),
        )
        return grain.DataLoader(
            data_source=self.train_source,
            sampler=sampler,
            operations=list(self._train_ops),
            worker_count=self._train_worker_count,
        )

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
        stats: dict[str, NormStats] | None = None,
        transforms: dict[str, Transform] | None = None,
        augmentations: dict[str, Augmentation] | None = None,
        worker_count: int = 0,
    ) -> GrainPipeline:
        """Build a GrainPipeline from pre-constructed DataSources.

        Parameters
        ----------
        train, valid, test : DataSource instances (windowed or full-seq).
        input_keys : Batch key names that are model inputs.
        target_keys : Batch key names that are model targets.
        bs : Batch size.
        seed : Shuffle seed for training data.
        stats : Pre-computed stats dict.  If None, stats are auto-computed
            by dispatching on reader type (SequenceReader, ScalarAttrReader, etc.).
        transforms : Per-key transforms applied per-sample before batching.
        augmentations : Per-key augmentations applied to training data only.
            Each augmentation receives ``(array, rng)`` and returns an array.
        worker_count : Number of DataLoader worker processes (0 = main process).
        """
        all_keys = tuple(input_keys) + tuple(target_keys)
        user_stats = stats or {}

        computed: dict[str, NormStats] = {}
        for key in all_keys:
            if key in user_stats:
                computed[key] = user_stats[key]
                continue
            transform = transforms.get(key) if transforms else None
            computed[key] = compute_stats(train, key, transform=transform)

        # Build DataLoader operations
        train_ops: list = []
        valid_ops: list = []
        test_ops: list = []

        if transforms:
            op = _MapOp(_apply_transforms(transforms))
            train_ops.append(op)
            valid_ops.append(op)
            test_ops.append(op)

        if augmentations:
            train_ops.append(_RandomMapOp(_apply_augmentations(augmentations)))

        train_ops.append(grain.transforms.Batch(bs, drop_remainder=True))
        valid_ops.append(grain.transforms.Batch(bs, drop_remainder=False))
        test_ops.append(grain.transforms.Batch(1, drop_remainder=False))

        # Valid/test use SequentialSampler — deterministic and reusable
        valid_dl = _make_sequential_loader(valid, valid_ops, worker_count=0)
        test_dl = _make_sequential_loader(test, test_ops, worker_count=0)

        n_train_batches = len(train) // bs

        return cls(
            valid=valid_dl,
            test=test_dl,
            stats=computed,
            input_keys=tuple(input_keys),
            target_keys=tuple(target_keys),
            train_source=train,
            valid_source=valid,
            test_source=test,
            n_train_batches=n_train_batches,
            _train_ops=train_ops,
            _train_worker_count=worker_count,
            _seed=seed,
        )


def _get_hdf_files(path: Path) -> list[str]:
    """Get sorted HDF5 files from a directory."""
    extensions = {".hdf5", ".h5"}
    return sorted(str(p) for p in path.rglob("*") if p.suffix in extensions)


def _get_split_files(dataset: Path | str, split: str) -> list[str]:
    """Get sorted HDF5 files for a specific split (train/valid/test)."""
    return _get_hdf_files(Path(dataset) / split)


def _needs_windowing(spec: ReaderSpec) -> bool:
    """Return True if the spec requires windowed iteration."""
    return isinstance(spec, (list, Feature))


def _collect_signal_names(specs: dict[str, ReaderSpec]) -> list[str]:
    """Collect unique signal names from all specs that read from the store."""
    signals: list[str] = []
    for spec in specs.values():
        if isinstance(spec, list):
            for s in spec:
                if s not in signals:
                    signals.append(s)
        elif isinstance(spec, Feature):
            for s in spec.signals:
                if s not in signals:
                    signals.append(s)
        # ScalarAttr reads HDF5 attrs, not datasets — no signals needed
    return signals


def _build_readers(
    specs: dict[str, ReaderSpec],
    store,
    files: list[str],
) -> dict[str, SequenceReader | ScalarAttrReader | FeatureReader]:
    """Create reader objects for each spec, for one split."""
    readers = {}
    for key, spec in specs.items():
        if isinstance(spec, list):
            readers[key] = SequenceReader(store, list(spec))
        elif isinstance(spec, ScalarAttr):
            readers[key] = ScalarAttrReader(files, spec.attrs)
        elif isinstance(spec, Feature):
            readers[key] = FeatureReader(store, spec.signals, spec.fn)
    return readers


def create_grain_dls(
    inputs: dict[str, ReaderSpec],
    targets: dict[str, ReaderSpec],
    dataset: Path | str,
    *,
    win_sz: int | None = None,
    stp_sz: int = 1,
    valid_stp_sz: int | None = None,
    bs: int = 64,
    seed: int = 42,
    preload: bool = False,
    resampling_factor: float | None = None,
    target_fs: float | None = None,
    fs_attr: str = "sampling_rate",
    resample_fn: ResampleFn | None = None,
    transforms: dict[str, Transform] | None = None,
    augmentations: dict[str, Augmentation] | None = None,
    worker_count: int = 0,
) -> GrainPipeline:
    """Create Grain data pipelines yielding raw (unnormalized) data.

    Parameters
    ----------
    inputs : dict mapping input batch key names to reader specs.
        E.g. ``{"u": ["u"]}`` for the common simulation case.
    targets : dict mapping target batch key names to reader specs.
        E.g. ``{"y": ["y"]}`` or ``{"y": ScalarAttr(["class"])}``.
    dataset : Path to dataset root containing train/valid/test splits.
    win_sz : Window size for windowed specs.  Required when any spec
        is a ``list[str]`` or ``Feature``.
    resampling_factor : float, optional
        Uniform resampling factor applied to all files.
    target_fs : float, optional
        Target sampling rate.  Per-file source rates are read from the
        HDF5 attribute named by *fs_attr*.
    fs_attr : str
        HDF5 root attribute containing the source sampling rate.
        Only used when *target_fs* is set.
    resample_fn : callable, optional
        Override the resampling algorithm (default: ``resample_interp``).
    transforms : dict mapping batch key names to transform functions, optional
        Per-key transforms applied to each sample before batching.
        Stats are computed on the transformed data.
    augmentations : dict mapping batch key names to augmentation functions, optional
        Per-key augmentations applied to training data only.
        Each augmentation is ``(ndarray, Generator) -> ndarray``.
        Applied after transforms.  Stats are computed on pre-augmentation data.
    worker_count : int
        Number of DataLoader worker processes (0 = main process only).
    """
    all_specs = {**inputs, **targets}

    # Determine if windowing is needed
    needs_win = any(_needs_windowing(s) for s in all_specs.values())
    if needs_win:
        if win_sz is None:
            raise ValueError(
                "win_sz is required when any spec needs windowing (list[str] or Feature)"
            )
    else:
        # All specs are ScalarAttr — file-level iteration
        win_sz = win_sz or 0  # unused, but set for type safety

    if valid_stp_sz is None:
        valid_stp_sz = win_sz

    # Validate transform keys
    if transforms:
        unknown = set(transforms) - set(all_specs)
        if unknown:
            raise ValueError(
                f"Transform keys {unknown} not found in inputs/targets "
                f"(available: {set(all_specs)})"
            )
    if augmentations:
        unknown = set(augmentations) - set(all_specs)
        if unknown:
            raise ValueError(
                f"Augmentation keys {unknown} not found in inputs/targets "
                f"(available: {set(all_specs)})"
            )

    # Collect signal names needed by the store (windowed/feature specs only)
    all_signals = _collect_signal_names(all_specs)

    dataset = Path(dataset)

    # Build separate mmap stores per split
    train_files = _get_split_files(dataset, "train")
    valid_files = _get_split_files(dataset, "valid")
    test_files = _get_split_files(dataset, "test")

    if all_signals:
        train_store = HDF5Store(train_files, all_signals, preload=preload)
        valid_store = HDF5Store(valid_files, all_signals, preload=preload)
        test_store = HDF5Store(test_files, all_signals, preload=preload)
    else:
        # Pure scalar — no signal datasets to read
        train_store = None
        valid_store = None
        test_store = None

    # Wrap with resampling if requested
    if train_store is not None:
        factor: float | Callable[[str], float] | None = resampling_factor
        if target_fs is not None:
            from .hdf5_store import read_hdf5_attr

            factor = lambda p: target_fs / float(read_hdf5_attr(p, fs_attr))  # noqa: E731
        if factor is not None:
            fn = resample_fn or resample_interp
            train_store = ResampledStore(train_store, factor, fn)
            valid_store = ResampledStore(valid_store, factor, fn)
            test_store = ResampledStore(test_store, factor, fn)

    # Build readers and DataSources
    train_readers = _build_readers(all_specs, train_store, train_files)
    valid_readers = _build_readers(all_specs, valid_store, valid_files)
    test_readers = _build_readers(all_specs, test_store, test_files)

    if needs_win:
        train_source = DataSource(train_store, train_readers, win_sz=win_sz, stp_sz=stp_sz)
        valid_source = DataSource(valid_store, valid_readers, win_sz=win_sz, stp_sz=valid_stp_sz)
        test_source = DataSource(test_store, test_readers)  # full sequence
    else:
        train_source = (
            DataSource(train_store, train_readers)
            if train_store
            else DataSource(_DummyStore(train_files), train_readers)
        )
        valid_source = (
            DataSource(valid_store, valid_readers)
            if valid_store
            else DataSource(_DummyStore(valid_files), valid_readers)
        )
        test_source = (
            DataSource(test_store, test_readers)
            if test_store
            else DataSource(_DummyStore(test_files), test_readers)
        )

    # Compute normalization stats from training data — one NormStats per key.
    # For transformed keys, stats are computed on the transformed output.
    stats: dict[str, NormStats] = {}
    for key in all_specs:
        transform = transforms.get(key) if transforms else None
        stats[key] = compute_stats(train_source, key, transform=transform)

    # Build DataLoader operations
    train_ops: list = []
    valid_ops: list = []
    test_ops: list = []

    if transforms:
        op = _MapOp(_apply_transforms(transforms))
        train_ops.append(op)
        valid_ops.append(op)
        test_ops.append(op)

    if augmentations:
        train_ops.append(_RandomMapOp(_apply_augmentations(augmentations)))

    train_ops.append(grain.transforms.Batch(bs, drop_remainder=True))
    valid_ops.append(grain.transforms.Batch(bs, drop_remainder=False))
    test_ops.append(grain.transforms.Batch(1, drop_remainder=False))

    # Valid/test use SequentialSampler — deterministic and reusable
    valid_dl = _make_sequential_loader(valid_source, valid_ops, worker_count=0)
    test_dl = _make_sequential_loader(test_source, test_ops, worker_count=0)

    n_train_batches = len(train_source) // bs

    return GrainPipeline(
        valid=valid_dl,
        test=test_dl,
        stats=stats,
        input_keys=tuple(inputs),
        target_keys=tuple(targets),
        train_source=train_source,
        valid_source=valid_source,
        test_source=test_source,
        n_train_batches=n_train_batches,
        _train_ops=train_ops,
        _train_worker_count=worker_count,
        _seed=seed,
    )


class _DummyStore:
    """Minimal store for pure-scalar pipelines (no signal datasets)."""

    def __init__(self, files: list[str]):
        self._paths = list(files)

    @property
    def paths(self):
        return self._paths

    def get_seq_len(self, path, signal=None):
        return 0

    def read_signals(self, path, signals, l_slc, r_slc):
        raise NotImplementedError("No signal datasets in a pure-scalar pipeline")


def create_simulation_dls(
    u: list[str],
    y: list[str],
    dataset: Path | str,
    *,
    win_sz: int = 100,
    **kw,
) -> GrainPipeline:
    """Shorthand for the common u->y simulation pipeline."""
    return create_grain_dls(
        inputs={"u": list(u)},
        targets={"y": list(y)},
        dataset=dataset,
        win_sz=win_sz,
        **kw,
    )
