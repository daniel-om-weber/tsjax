"""Integration tests for the data pipeline: stats, HDF5 reading, windowing, batching."""

from functools import partial

import h5py
import numpy as np
import pytest


def _make_source(dataset_path, signals):
    """Build a minimal FileSource for stats tests."""
    from tsjax import FileSource, HDF5Store

    train_files = sorted(str(p) for p in (dataset_path / "train").rglob("*.hdf5"))
    store = HDF5Store(train_files, signals)
    return FileSource(store, {s: [s] for s in signals})


def _make_loader(source, bs=4):
    """Wrap a DataSource in a sequential IterDataset for stats tests."""
    import grain

    return grain.MapDataset.source(source).batch(bs, drop_remainder=False).to_iter_dataset()


def test_norm_stats_shape_and_finite(dataset_path):
    """compute_stats returns finite float32 arrays with one element per signal."""
    from tsjax import compute_stats

    source = _make_source(dataset_path, ["u"])
    dl = _make_loader(source)
    result = compute_stats(dl, ["u"], n_batches=100)
    ns = result["u"]
    assert ns.mean.shape == (1,)
    assert ns.std.shape == (1,)
    assert ns.mean.dtype == np.float32
    assert np.all(np.isfinite(ns.mean))
    assert np.all(ns.std > 0)


def test_norm_stats_match_manual_computation(dataset_path):
    """Stats should match a direct h5py mean/std calculation."""
    from tsjax import compute_stats

    train_files = sorted(str(p) for p in (dataset_path / "train").rglob("*.hdf5"))
    source = _make_source(dataset_path, ["u"])
    dl = _make_loader(source)
    result = compute_stats(dl, ["u"], n_batches=1000)
    ns = result["u"]

    # Manual computation
    all_data = []
    for f in train_files:
        with h5py.File(f, "r") as hf:
            all_data.append(hf["u"][:])
    data = np.concatenate(all_data)
    expected_mean = np.mean(data).astype(np.float32)
    expected_std = np.std(data).astype(np.float32)

    np.testing.assert_allclose(ns.mean[0], expected_mean, atol=1e-5)
    np.testing.assert_allclose(ns.std[0], expected_std, atol=1e-5)


def test_hdf5_store_reads_match_h5py(dataset_path):
    """HDF5Store slicing should match direct h5py reads."""
    from tsjax import HDF5Store

    train_files = sorted(str(p) for p in (dataset_path / "train").rglob("*.hdf5"))
    index = HDF5Store(train_files, ["u", "y"])

    path = train_files[0]
    with h5py.File(path, "r") as f:
        expected = f["u"][10:50].astype(np.float32)

    result = index.read_slice(path, "u", 10, 50)
    np.testing.assert_array_equal(result, expected)


def test_pipeline_train_batch_shapes(pipeline):
    """Train batches should be (bs, win_sz, n_signals)."""
    batch = next(iter(pipeline.train))
    assert batch["u"].shape == (4, 20, 1)
    assert batch["y"].shape == (4, 20, 1)


def test_multiworker_train_loader(dataset_path):
    """Train loader with worker_count>0 should produce correct batches."""
    from tsjax import create_simulation_dls

    pl = create_simulation_dls(
        u=["u"],
        y=["y"],
        dataset=dataset_path,
        win_sz=20,
        stp_sz=10,
        bs=4,
        seed=42,
        worker_count=2,
    )
    train_iter = iter(pl.train)
    batches = [next(train_iter) for _ in range(pl.n_train_batches)]
    assert len(batches) > 0
    for batch in batches:
        assert batch["u"].shape == (4, 20, 1)
        assert batch["y"].shape == (4, 20, 1)
        assert np.all(np.isfinite(batch["u"]))
        assert np.all(np.isfinite(batch["y"]))


def test_pipeline_has_norm_stats(pipeline):
    """Pipeline should expose finite norm stats with correct length."""
    assert len(pipeline.stats["u"].mean) == 1
    assert len(pipeline.stats["u"].std) == 1
    assert len(pipeline.stats["y"].mean) == 1
    assert len(pipeline.stats["y"].std) == 1
    assert np.all(np.isfinite(pipeline.stats["u"].mean))
    assert np.all(pipeline.stats["u"].std > 0)


def test_pipeline_has_key_metadata(pipeline):
    """Pipeline should expose input_keys and target_keys."""
    assert pipeline.input_keys == ("u",)
    assert pipeline.target_keys == ("y",)


def test_test_split_full_sequences(pipeline):
    """Test split should yield full-length sequences (longer than win_sz)."""
    batch = next(iter(pipeline.test))
    assert batch["u"].shape[0] == 1  # batch size 1
    assert batch["u"].shape[1] > 20  # full sequence, not windowed
    assert batch["u"].shape[2] == 1


def test_window_count_matches_formula(dataset_path):
    """Window count should match: (seq_len - win_sz) // stp_sz + 1."""
    from tsjax import HDF5Store, WindowedSource

    train_files = sorted(str(p) for p in (dataset_path / "train").rglob("*.hdf5"))
    store = HDF5Store(train_files, ["u", "y"])
    source = WindowedSource(store, {"u": ["u"]}, win_sz=20, stp_sz=10)

    # Verify against formula
    expected_total = 0
    for path in train_files:
        seq_len = store.get_seq_len(path, "u")
        expected_total += max(0, (seq_len - 20) // 10 + 1)

    assert len(source) == expected_total


# ---------------------------------------------------------------------------
# Synthetic HDF5 fixtures for new reader types
# ---------------------------------------------------------------------------


@pytest.fixture()
def scalar_hdf5_files(tmp_path):
    """Create HDF5 files with signal datasets and scalar attributes."""
    files = []
    rng = np.random.default_rng(42)
    for i in range(3):
        path = str(tmp_path / f"file_{i}.hdf5")
        with h5py.File(path, "w") as f:
            f.create_dataset("u", data=rng.standard_normal(100).astype(np.float32))
            f.create_dataset("y", data=rng.standard_normal(100).astype(np.float32))
            f.attrs["mass"] = 1.0 + i
            f.attrs["stiffness"] = 10.0 + i * 2
            f.attrs["class_label"] = float(i % 2)
        files.append(path)
    return files


# ---------------------------------------------------------------------------
# scalar_attrs tests
# ---------------------------------------------------------------------------


class TestScalarAttrs:
    def test_reads_single_attr(self, scalar_hdf5_files):
        from tsjax.data.sources import scalar_attrs

        reader = scalar_attrs(scalar_hdf5_files, ["mass"])
        result = reader(scalar_hdf5_files[0], 0, 0)
        assert result.shape == (1,)
        assert result[0] == pytest.approx(1.0)

    def test_reads_multiple_attrs(self, scalar_hdf5_files):
        from tsjax.data.sources import scalar_attrs

        reader = scalar_attrs(scalar_hdf5_files, ["mass", "stiffness"])
        result = reader(scalar_hdf5_files[1], 0, 0)
        assert result.shape == (2,)
        assert result[0] == pytest.approx(2.0)
        assert result[1] == pytest.approx(12.0)

    def test_ignores_slices(self, scalar_hdf5_files):
        from tsjax.data.sources import scalar_attrs

        reader = scalar_attrs(scalar_hdf5_files, ["mass"])
        r1 = reader(scalar_hdf5_files[0], 0, 0)
        r2 = reader(scalar_hdf5_files[0], 10, 50)
        np.testing.assert_array_equal(r1, r2)

    def test_dtype_is_float32(self, scalar_hdf5_files):
        from tsjax.data.sources import scalar_attrs

        reader = scalar_attrs(scalar_hdf5_files, ["mass"])
        result = reader(scalar_hdf5_files[0], 0, 0)
        assert result.dtype == np.float32


# ---------------------------------------------------------------------------
# signal_feature tests
# ---------------------------------------------------------------------------


class TestSignalFeature:
    def test_applies_mean_reduction(self, scalar_hdf5_files):
        from tsjax import HDF5Store
        from tsjax.data.sources import signal_feature

        store = HDF5Store(scalar_hdf5_files, ["u"], preload=True)
        fn = partial(np.mean, axis=0)
        reader = signal_feature(store, ["u"], fn)
        result = reader(scalar_hdf5_files[0], 10, 30)
        assert result.shape == (1,)

        # Verify manually
        expected = np.mean(store.read_signals(scalar_hdf5_files[0], ["u"], 10, 30), axis=0)
        np.testing.assert_allclose(result, expected.astype(np.float32), atol=1e-6)

    def test_multi_signal_reduction(self, scalar_hdf5_files):
        from tsjax import HDF5Store
        from tsjax.data.sources import signal_feature

        store = HDF5Store(scalar_hdf5_files, ["u", "y"], preload=True)
        fn = partial(np.mean, axis=0)
        reader = signal_feature(store, ["u", "y"], fn)
        result = reader(scalar_hdf5_files[0], 0, 50)
        assert result.shape == (2,)
        assert result.dtype == np.float32


# ---------------------------------------------------------------------------
# compute_scalar_stats tests
# ---------------------------------------------------------------------------


class TestComputeScalarStats:
    @staticmethod
    def _make_scalar_loader(files, attrs):
        from tsjax import HDF5Store
        from tsjax.data.sources import FileSource, scalar_attrs

        reader = scalar_attrs(files, attrs)
        store = HDF5Store(files, [])
        source = FileSource(store, {"x": reader})
        return _make_loader(source, bs=len(files))

    def test_shape_and_dtype(self, scalar_hdf5_files):
        from tsjax.data.stats import compute_stats

        dl = self._make_scalar_loader(scalar_hdf5_files, ["mass"])
        result = compute_stats(dl, ["x"], n_batches=10)
        ns = result["x"]
        assert ns.mean.shape == (1,)
        assert ns.std.shape == (1,)
        assert ns.mean.dtype == np.float32
        assert ns.std.dtype == np.float32

    def test_matches_manual(self, scalar_hdf5_files):
        from tsjax.data.stats import compute_stats

        dl = self._make_scalar_loader(scalar_hdf5_files, ["mass"])
        result = compute_stats(dl, ["x"], n_batches=10)
        ns = result["x"]
        vals = np.array([1.0, 2.0, 3.0])
        np.testing.assert_allclose(ns.mean[0], np.mean(vals), atol=1e-6)
        np.testing.assert_allclose(ns.std[0], np.std(vals), atol=1e-6)

    def test_multi_attr(self, scalar_hdf5_files):
        from tsjax.data.stats import compute_stats

        dl = self._make_scalar_loader(scalar_hdf5_files, ["mass", "stiffness"])
        result = compute_stats(dl, ["x"], n_batches=10)
        ns = result["x"]
        assert ns.mean.shape == (2,)
        assert ns.std.shape == (2,)

    def test_empty_attrs(self, scalar_hdf5_files):
        from tsjax.data.stats import compute_stats

        dl = self._make_scalar_loader(scalar_hdf5_files, [])
        result = compute_stats(dl, ["x"], n_batches=10)
        ns = result["x"]
        assert ns.mean.shape == (0,)
        assert ns.std.shape == (0,)


# ---------------------------------------------------------------------------
# Manual pipeline with non-signal readers
# ---------------------------------------------------------------------------


class TestManualPipeline:
    def test_pipeline_with_scalar_target(self, tmp_path):
        """Manual pipeline with scalar_attrs target should produce scalar batches."""
        from tsjax import GrainPipeline, HDF5Store, WindowedSource
        from tsjax.data.sources import FileSource, scalar_attrs

        rng = np.random.default_rng(123)
        split_files = {}
        for split in ("train", "valid", "test"):
            d = tmp_path / "split" / split
            d.mkdir(parents=True)
            files = []
            for i in range(2):
                path = str(d / f"file_{i}.hdf5")
                with h5py.File(path, "w") as f:
                    f.create_dataset("u", data=rng.standard_normal(100).astype(np.float32))
                    f.attrs["class_label"] = float(i % 2)
                files.append(path)
            split_files[split] = files

        def make_source(split, windowed=True):
            files = split_files[split]
            store = HDF5Store(files, ["u"])
            specs = {"u": ["u"], "y": scalar_attrs(files, ["class_label"])}
            if windowed:
                return WindowedSource(store, specs, win_sz=20, stp_sz=20)
            return FileSource(store, specs)

        pl = GrainPipeline.from_sources(
            make_source("train"),
            make_source("valid"),
            make_source("test", windowed=False),
            input_keys=("u",),
            target_keys=("y",),
            bs=2,
            seed=42,
        )
        assert pl.input_keys == ("u",)
        assert pl.target_keys == ("y",)
        batch = next(iter(pl.train))
        assert batch["u"].shape == (2, 20, 1)  # windowed
        assert batch["y"].shape == (2, 1)  # scalar

    def test_pipeline_with_feature_target(self, tmp_path):
        """Manual pipeline with signal_feature target should produce reduced batches."""
        from tsjax import GrainPipeline, HDF5Store, WindowedSource
        from tsjax.data.sources import FileSource, signal_feature

        rng = np.random.default_rng(456)
        split_files = {}
        for split in ("train", "valid", "test"):
            d = tmp_path / "split" / split
            d.mkdir(parents=True)
            files = []
            for i in range(2):
                path = str(d / f"file_{i}.hdf5")
                with h5py.File(path, "w") as f:
                    f.create_dataset("u", data=rng.standard_normal(100).astype(np.float32))
                    f.create_dataset("y", data=rng.standard_normal(100).astype(np.float32))
                files.append(path)
            split_files[split] = files

        fn = partial(np.mean, axis=0)

        def make_source(split, windowed=True):
            files = split_files[split]
            store = HDF5Store(files, ["u", "y"])
            specs = {"u": ["u"], "y": signal_feature(store, ["y"], fn)}
            if windowed:
                return WindowedSource(store, specs, win_sz=20, stp_sz=20)
            return FileSource(store, specs)

        pl = GrainPipeline.from_sources(
            make_source("train"),
            make_source("valid"),
            make_source("test", windowed=False),
            input_keys=("u",),
            target_keys=("y",),
            bs=2,
            seed=42,
        )
        batch = next(iter(pl.train))
        assert batch["u"].shape == (2, 20, 1)  # windowed
        assert batch["y"].shape == (2, 1)  # feature-reduced


# ---------------------------------------------------------------------------
# Numerical stability regression test
# ---------------------------------------------------------------------------


def test_stats_numerically_stable_large_offset(tmp_path):
    """Variance computation must be stable for large-offset, small-variance data.

    The naive formula std = sqrt(E[x^2] - E[x]^2) produces catastrophic
    cancellation here (negative variance -> NaN).  numpy handles this correctly.
    """
    from tsjax import FileSource, HDF5Store, compute_stats

    # Data centered at 1e6 with std=0.01 -- triggers cancellation in naive formula
    rng = np.random.default_rng(42)
    offset = 1e6
    true_std = 0.01
    n_samples = 10_000

    path = tmp_path / "large_offset.hdf5"
    data = (rng.standard_normal(n_samples) * true_std + offset).astype(np.float32)
    with h5py.File(str(path), "w") as f:
        f.create_dataset("y", data=data)

    store = HDF5Store([str(path)], ["y"])
    source = FileSource(store, {"y": ["y"]})
    dl = _make_loader(source, bs=1000)
    result = compute_stats(dl, ["y"], n_batches=100)
    ns = result["y"]

    expected_mean = np.mean(data)
    expected_std = np.std(data)

    assert np.all(np.isfinite(ns.mean)), "mean contains NaN/Inf"
    assert np.all(np.isfinite(ns.std)), "std contains NaN/Inf"
    assert ns.std[0] > 0, "std should be positive"
    np.testing.assert_allclose(ns.mean[0], expected_mean, rtol=1e-5)
    np.testing.assert_allclose(ns.std[0], expected_std, rtol=1e-2)
