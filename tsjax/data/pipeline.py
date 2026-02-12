"""Factory function to create Grain data pipelines yielding raw data."""

from __future__ import annotations

import functools
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
)
from .resample import ResampledStore, ResampleFn, resample_interp
from .sources import (
    FileSource,
    WindowedSource,
)
from .stats import (
    NormStats,
    compute_stats,
)
from .store import SignalStore


@dataclass
class GrainPipeline:
    """Container for train/valid/test Grain IterDatasets with norm stats."""

    train: grain.IterDataset
    valid: grain.IterDataset
    test: grain.IterDataset
    input_keys: tuple[str, ...]
    target_keys: tuple[str, ...]
    train_source: WindowedSource | FileSource
    valid_source: WindowedSource | FileSource
    test_source: WindowedSource | FileSource
    bs: int
    _stats_batches: int = field(default=10, repr=False)
    _n_train_batches_override: int | None = field(default=None, init=False, repr=False)

    @property
    def n_train_batches(self) -> int:
        if self._n_train_batches_override is not None:
            return self._n_train_batches_override
        return len(self.train_source) // self.bs

    @n_train_batches.setter
    def n_train_batches(self, value: int) -> None:
        self._n_train_batches_override = value

    @functools.cached_property
    def stats(self) -> dict[str, NormStats]:
        """Lazily compute normalization stats on first access."""
        return compute_stats(self.valid, n_batches=self._stats_batches)

    @classmethod
    def from_sources(
        cls,
        train: WindowedSource | FileSource,
        valid: WindowedSource | FileSource,
        test: WindowedSource | FileSource,
        *,
        input_keys: tuple[str, ...],
        target_keys: tuple[str, ...],
        bs: int = 64,
        seed: int = 42,
        transforms: dict[str, Transform] | None = None,
        augmentations: dict[str, Augmentation] | None = None,
        worker_count: int = 0,
        stats_batches: int = 10,
    ) -> GrainPipeline:
        """Build a GrainPipeline from pre-constructed sources.

        Parameters
        ----------
        train, valid, test : WindowedSource or FileSource instances.
        input_keys : Batch key names that are model inputs.
        target_keys : Batch key names that are model targets.
        bs : Batch size.
        seed : Shuffle seed for training data.
        transforms : Per-key transforms applied per-sample before batching.
        augmentations : Per-key augmentations applied to training data only.
            Each augmentation receives ``(array, rng)`` and returns an array.
        worker_count : Number of worker processes for training (0 = main process).
        stats_batches : Number of batches to sample for auto-computing stats.
        """
        transform_fn = _apply_transforms(transforms) if transforms else None
        augment_fn = _apply_augmentations(augmentations) if augmentations else None

        # Train: infinite shuffled with optional augmentations
        train_ds = grain.MapDataset.source(train).seed(seed).shuffle().repeat(None)
        if transform_fn:
            train_ds = train_ds.map(transform_fn)
        if augment_fn:
            train_ds = train_ds.random_map(augment_fn)
        train_iter = train_ds.batch(bs, drop_remainder=True).to_iter_dataset()
        if worker_count > 0:
            train_iter = train_iter.mp_prefetch(
                grain.MultiprocessingOptions(num_workers=worker_count)
            )

        # Valid: sequential, no augmentations
        valid_ds = grain.MapDataset.source(valid)
        if transform_fn:
            valid_ds = valid_ds.map(transform_fn)
        valid_iter = valid_ds.batch(bs, drop_remainder=False).to_iter_dataset()

        # Test: sequential, batch_size=1
        test_ds = grain.MapDataset.source(test)
        if transform_fn:
            test_ds = test_ds.map(transform_fn)
        test_iter = test_ds.batch(1, drop_remainder=False).to_iter_dataset()

        return cls(
            train=train_iter,
            valid=valid_iter,
            test=test_iter,
            input_keys=tuple(input_keys),
            target_keys=tuple(target_keys),
            train_source=train,
            valid_source=valid,
            test_source=test,
            bs=bs,
            _stats_batches=stats_batches,
        )


def _get_hdf_files(path: Path) -> list[str]:
    """Get sorted HDF5 files from a directory."""
    extensions = {".hdf5", ".h5"}
    return sorted(str(p) for p in path.rglob("*") if p.suffix in extensions)


def _get_split_files(dataset: Path | str, split: str) -> list[str]:
    """Get sorted HDF5 files for a specific split (train/valid/test)."""
    return _get_hdf_files(Path(dataset) / split)


def create_grain_dls(
    inputs: dict[str, list[str]],
    targets: dict[str, list[str]],
    dataset: Path | str,
    *,
    win_sz: int,
    stp_sz: int = 1,
    valid_stp_sz: int | None = None,
    bs: int = 64,
    seed: int = 42,
    preload: bool = False,
    store_factory: Callable[[list[str], list[str]], SignalStore] | None = None,
    resampling_factor: float | None = None,
    target_fs: float | None = None,
    fs_attr: str = "sampling_rate",
    resample_fn: ResampleFn | None = None,
    transforms: dict[str, Transform] | None = None,
    augmentations: dict[str, Augmentation] | None = None,
    worker_count: int = 0,
    stats_batches: int = 10,
) -> GrainPipeline:
    """Create Grain data pipelines yielding raw (unnormalized) data.

    Parameters
    ----------
    inputs : dict mapping input batch key names to signal lists.
        E.g. ``{"u": ["u"]}`` for the common simulation case.
    targets : dict mapping target batch key names to signal lists.
        E.g. ``{"y": ["y"]}``.
    dataset : Path to dataset root containing train/valid/test splits.
    win_sz : Window size (number of time steps per sample).
    store_factory : callable, optional
        ``(paths, signal_names) -> SignalStore``.  Defaults to
        ``HDF5Store(paths, signal_names, preload=preload)``.
        Use this to plug in custom stores (e.g. CSV, Parquet).
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
    augmentations : dict mapping batch key names to augmentation functions, optional
        Per-key augmentations applied to training data only.
        Each augmentation is ``(ndarray, Generator) -> ndarray``.
        Applied after transforms.
    worker_count : int
        Number of DataLoader worker processes (0 = main process only).
    stats_batches : int
        Number of batches to sample for auto-computing stats.
    """
    all_specs = {**inputs, **targets}

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

    # Collect unique signal names
    seen: set[str] = set()
    all_signals: list[str] = []
    for sig_list in all_specs.values():
        for s in sig_list:
            if s not in seen:
                seen.add(s)
                all_signals.append(s)

    dataset = Path(dataset)
    train_files = _get_split_files(dataset, "train")
    valid_files = _get_split_files(dataset, "valid")
    test_files = _get_split_files(dataset, "test")

    # Build stores
    _make_store = store_factory or (lambda paths, sigs: HDF5Store(paths, sigs, preload=preload))
    train_store: SignalStore = _make_store(train_files, all_signals)
    valid_store: SignalStore = _make_store(valid_files, all_signals)
    test_store: SignalStore = _make_store(test_files, all_signals)

    # Wrap with resampling if requested
    factor: float | Callable[[str], float] | None = resampling_factor
    if target_fs is not None:
        from .hdf5_store import read_hdf5_attr

        factor = lambda p: target_fs / float(read_hdf5_attr(p, fs_attr))  # noqa: E731
    if factor is not None:
        fn = resample_fn or resample_interp
        train_store = ResampledStore(train_store, factor, fn)
        valid_store = ResampledStore(valid_store, factor, fn)
        test_store = ResampledStore(test_store, factor, fn)

    # Build sources directly from signal specs
    train_source = WindowedSource(train_store, all_specs, win_sz=win_sz, stp_sz=stp_sz)
    valid_source = WindowedSource(valid_store, all_specs, win_sz=win_sz, stp_sz=valid_stp_sz)
    test_source = FileSource(test_store, all_specs)

    return GrainPipeline.from_sources(
        train_source,
        valid_source,
        test_source,
        input_keys=tuple(inputs),
        target_keys=tuple(targets),
        bs=bs,
        seed=seed,
        transforms=transforms,
        augmentations=augmentations,
        worker_count=worker_count,
        stats_batches=stats_batches,
    )


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
