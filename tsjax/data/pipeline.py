"""Factory function to create Grain data pipelines yielding raw data."""

from __future__ import annotations

import functools
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import grain
import numpy as np

from .hdf5_store import HDF5Store, discover_split_files
from .resample import ResampledStore, ResampleFn, resample_interp
from .sampling import WeightedMapDataset
from .sources import WindowedSource
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
    n_train_batches: int
    signal_names: dict[str, list[str]] = field(default_factory=dict)

    def __hash__(self) -> int:
        return id(self)

    @functools.cache
    def stats(self, n_batches: int = 10) -> dict[str, NormStats]:
        """Compute normalization stats from the validation set (cached)."""
        return compute_stats(self.valid, n_batches=n_batches)

    @classmethod
    def from_sources(
        cls,
        train: grain.RandomAccessDataSource,
        valid: grain.RandomAccessDataSource,
        test: grain.RandomAccessDataSource,
        *,
        input_keys: tuple[str, ...],
        target_keys: tuple[str, ...],
        bs: int = 64,
        seed: int = 42,
        weights: np.ndarray | Sequence[float] | None = None,
        transform: Callable[[dict[str, np.ndarray]], dict[str, np.ndarray]] | None = None,
        augmentation: (
            Callable[[dict[str, np.ndarray], np.random.Generator], dict[str, np.ndarray]] | None
        ) = None,
        worker_count: int = 0,
    ) -> GrainPipeline:
        """Build a GrainPipeline from pre-constructed sources.

        Parameters
        ----------
        train, valid, test : Any grain-compatible random-access sources
            (``__len__`` + ``__getitem__``).
        input_keys : Batch key names that are model inputs.
        target_keys : Batch key names that are model targets.
        bs : Batch size.
        seed : Shuffle seed for training data.
        weights : Per-element sampling weights for the training set.
            Higher-weight elements appear more often per epoch.  When
            ``None`` (default), all elements are sampled uniformly.
        transform : Optional function applied per-sample before batching
            (all splits). Signature: ``(sample_dict) -> sample_dict``.
        augmentation : Optional function applied to training data only.
            Signature: ``(sample_dict, rng) -> sample_dict``.
        worker_count : Number of worker processes for training (0 = main process).
        """
        # Train: infinite shuffled with optional augmentations
        train_ds = grain.MapDataset.source(train)
        if weights is not None:
            train_ds = train_ds.pipe(WeightedMapDataset, weights=weights)
        n_train = len(train_ds)
        train_ds = train_ds.seed(seed).shuffle().repeat(None)
        if transform:
            train_ds = train_ds.map(transform)
        if augmentation:
            train_ds = train_ds.random_map(augmentation)
        train_iter = train_ds.batch(bs, drop_remainder=True).to_iter_dataset()
        if worker_count > 0:
            train_iter = train_iter.mp_prefetch(
                grain.MultiprocessingOptions(num_workers=worker_count)
            )

        # Valid: sequential, no augmentations
        valid_ds = grain.MapDataset.source(valid)
        if transform:
            valid_ds = valid_ds.map(transform)
        valid_iter = valid_ds.batch(bs, drop_remainder=False).to_iter_dataset()

        # Test: sequential, batch_size=1
        test_ds = grain.MapDataset.source(test)
        if transform:
            test_ds = test_ds.map(transform)
        test_iter = test_ds.batch(1, drop_remainder=False).to_iter_dataset()

        return cls(
            train=train_iter,
            valid=valid_iter,
            test=test_iter,
            input_keys=tuple(input_keys),
            target_keys=tuple(target_keys),
            n_train_batches=n_train // bs,
            signal_names=getattr(train, "signal_names", {}),
        )


def create_grain_dls(
    inputs: dict[str, list[str]],
    targets: dict[str, list[str]],
    dataset: Path | str | None = None,
    *,
    win_sz: int,
    stp_sz: int = 1,
    valid_stp_sz: int | None = None,
    bs: int = 64,
    seed: int = 42,
    preload: bool = False,
    train_files: Sequence[str] | None = None,
    valid_files: Sequence[str] | None = None,
    test_files: Sequence[str] | None = None,
    store_factory: Callable[[list[str], list[str]], SignalStore] | None = None,
    resampling_factor: float | None = None,
    target_fs: float | None = None,
    fs_attr: str = "sampling_rate",
    resample_fn: ResampleFn | None = None,
    weights: np.ndarray | Sequence[float] | Callable[[WindowedSource], np.ndarray] | None = None,
    transform: Callable[[dict[str, np.ndarray]], dict[str, np.ndarray]] | None = None,
    augmentation: (
        Callable[[dict[str, np.ndarray], np.random.Generator], dict[str, np.ndarray]] | None
    ) = None,
    worker_count: int = 0,
) -> GrainPipeline:
    """Create Grain data pipelines yielding raw (unnormalized) data.

    Provide either ``dataset`` (auto-discovers files from ``train/``,
    ``valid/``, ``test/`` subdirectories) **or** explicit file lists via
    ``train_files``, ``valid_files``, ``test_files``.

    Parameters
    ----------
    inputs : dict mapping input batch key names to signal lists.
        E.g. ``{"u": ["u"]}`` for the common simulation case.
    targets : dict mapping target batch key names to signal lists.
        E.g. ``{"y": ["y"]}``.
    dataset : Path to dataset root containing train/valid/test splits.
        Mutually exclusive with *train_files*/*valid_files*/*test_files*.
    win_sz : Window size (number of time steps per sample).
    train_files, valid_files, test_files : explicit file path lists.
        When provided, *dataset* must be ``None``.  Use
        :func:`discover_split_files` to get these from a standard layout,
        then filter/merge before passing them here.
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
    weights : array-like or callable, optional
        Per-element sampling weights for the training set.  Higher-weight
        elements appear more often per epoch.  When a callable, it is
        called with the train ``WindowedSource`` to compute the weights
        (e.g. ``weights=uniform_file_weights``).
    transform : callable, optional
        Per-sample transform applied before batching (all splits).
        Signature: ``(sample_dict) -> sample_dict``.
    augmentation : callable, optional
        Training-only augmentation applied after transforms.
        Signature: ``(sample_dict, rng) -> sample_dict``.
    worker_count : int
        Number of DataLoader worker processes (0 = main process only).
    """
    all_specs = {**inputs, **targets}

    if valid_stp_sz is None:
        valid_stp_sz = win_sz

    # Collect unique signal names
    seen: set[str] = set()
    all_signals: list[str] = []
    for sig_list in all_specs.values():
        for s in sig_list:
            if s not in seen:
                seen.add(s)
                all_signals.append(s)

    # Resolve file lists
    has_explicit = train_files is not None or valid_files is not None or test_files is not None
    if has_explicit and dataset is not None:
        raise ValueError("Provide either 'dataset' or explicit file lists, not both.")
    if has_explicit:
        if train_files is None or valid_files is None or test_files is None:
            raise ValueError(
                "All three file lists (train_files, valid_files, test_files) must be provided."
            )
        _train_files = list(train_files)
        _valid_files = list(valid_files)
        _test_files = list(test_files)
    elif dataset is not None:
        _train_files, _valid_files, _test_files = discover_split_files(dataset)
    else:
        raise ValueError("Provide either 'dataset' or explicit file lists.")

    # Build stores
    _make_store = store_factory or (lambda paths, sigs: HDF5Store(paths, sigs, preload=preload))
    train_store: SignalStore = _make_store(_train_files, all_signals)
    valid_store: SignalStore = _make_store(_valid_files, all_signals)
    test_store: SignalStore = _make_store(_test_files, all_signals)

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
    test_source = WindowedSource(test_store, all_specs)

    # Resolve callable weights (e.g. uniform_file_weights)
    resolved_weights = weights(train_source) if callable(weights) else weights

    return GrainPipeline.from_sources(
        train_source,
        valid_source,
        test_source,
        input_keys=tuple(inputs),
        target_keys=tuple(targets),
        bs=bs,
        seed=seed,
        weights=resolved_weights,
        transform=transform,
        augmentation=augmentation,
        worker_count=worker_count,
    )


def create_simulation_dls(
    u: list[str],
    y: list[str],
    dataset: Path | str | None = None,
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
