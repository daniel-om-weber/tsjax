"""Factory function to create Grain data pipelines yielding raw data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import grain
import numpy as np

from .hdf5_index import HDF5MmapIndex
from .sources import FullSequenceSource, WindowedSource
from .stats import compute_norm_stats


@dataclass
class GrainPipeline:
    """Container for train/valid/test Grain datasets with norm stats."""

    train: grain.MapDataset
    valid: grain.MapDataset
    test: grain.MapDataset
    u_mean: "np.ndarray"
    u_std: "np.ndarray"
    y_mean: "np.ndarray"
    y_std: "np.ndarray"
    train_source: WindowedSource
    valid_source: WindowedSource
    test_source: FullSequenceSource


def _get_hdf_files(path: Path) -> list[str]:
    """Get sorted HDF5 files from a directory."""
    extensions = {".hdf5", ".h5"}
    return sorted(str(p) for p in path.rglob("*") if p.suffix in extensions)


def _get_split_files(dataset: Path | str, split: str) -> list[str]:
    """Get sorted HDF5 files for a specific split (train/valid/test)."""
    return _get_hdf_files(Path(dataset) / split)


def create_grain_dls(
    u: list[str],
    y: list[str],
    dataset: Path | str,
    win_sz: int = 100,
    stp_sz: int = 1,
    valid_stp_sz: int | None = None,
    bs: int = 64,
    seed: int = 42,
    preload: bool = False,
) -> GrainPipeline:
    """Create Grain data pipelines yielding raw (unnormalized) data.

    Norm stats are stored on the pipeline for use by the model.
    """
    if valid_stp_sz is None:
        valid_stp_sz = win_sz

    all_signals = list(u) + list(y)
    dataset = Path(dataset)

    # Build separate mmap indices per split
    train_files = _get_split_files(dataset, "train")
    valid_files = _get_split_files(dataset, "valid")
    test_files = _get_split_files(dataset, "test")

    train_index = HDF5MmapIndex(train_files, all_signals, preload=preload)
    valid_index = HDF5MmapIndex(valid_files, all_signals, preload=preload)
    test_index = HDF5MmapIndex(test_files, all_signals, preload=preload)

    # Compute normalization stats from training data (input + output signals)
    u_mean, u_std = compute_norm_stats(train_files, list(u))
    y_mean, y_std = compute_norm_stats(train_files, list(y))

    # Build sources
    train_source = WindowedSource(train_index, win_sz, stp_sz, list(u), list(y))
    valid_source = WindowedSource(valid_index, win_sz, valid_stp_sz, list(u), list(y))
    test_source = FullSequenceSource(test_index, list(u), list(y))

    # Build pipelines — yield raw data, no normalization
    train_ds = (
        grain.MapDataset.source(train_source).shuffle(seed=seed).batch(bs, drop_remainder=True)
    )

    valid_ds = grain.MapDataset.source(valid_source).batch(bs, drop_remainder=False)

    test_ds = grain.MapDataset.source(test_source).batch(1, drop_remainder=False)

    return GrainPipeline(
        train=train_ds,
        valid=valid_ds,
        test=test_ds,
        u_mean=u_mean,
        u_std=u_std,
        y_mean=y_mean,
        y_std=y_std,
        train_source=train_source,
        valid_source=valid_source,
        test_source=test_source,
    )
