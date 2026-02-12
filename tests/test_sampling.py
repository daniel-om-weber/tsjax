"""Tests for weighted sampling: WeightedMapDataset and uniform_file_weights."""

import grain
import h5py
import numpy as np
import pytest

# ---------------------------------------------------------------------------
# WeightedMapDataset unit tests
# ---------------------------------------------------------------------------


class TestWeightedMapDataset:
    def test_uniform_weights_same_length(self):
        """Equal weights should produce the same length as the parent."""
        from tsjax.data.sampling import WeightedMapDataset

        parent = grain.MapDataset.source(list(range(5)))
        weights = [1.0, 1.0, 1.0, 1.0, 1.0]
        ds = parent.pipe(WeightedMapDataset, weights=weights)
        assert len(ds) == 5

    def test_weighted_expands_length(self):
        """Higher weights should expand the dataset length."""
        from tsjax.data.sampling import WeightedMapDataset

        parent = grain.MapDataset.source(list(range(3)))
        # weights [3, 1, 1]: element 0 appears 3x, others 1x → total 5
        ds = parent.pipe(WeightedMapDataset, weights=[3.0, 1.0, 1.0])
        assert len(ds) == 5

    def test_element_frequencies_match_weights(self):
        """Over one epoch, element counts should be proportional to weights."""
        from tsjax.data.sampling import WeightedMapDataset

        parent = grain.MapDataset.source(list(range(4)))
        weights = [4.0, 2.0, 1.0, 1.0]
        ds = parent.pipe(WeightedMapDataset, weights=weights)

        # Collect all elements in one epoch
        elements = [ds[i] for i in range(len(ds))]
        counts = {v: elements.count(v) for v in range(4)}
        assert counts[0] == 4
        assert counts[1] == 2
        assert counts[2] == 1
        assert counts[3] == 1

    def test_shuffle_preserves_distribution(self):
        """Shuffle + WeightedMapDataset should produce correct frequencies."""
        from tsjax.data.sampling import WeightedMapDataset

        parent = grain.MapDataset.source(list(range(4)))
        weights = [4.0, 2.0, 1.0, 1.0]
        ds = parent.pipe(WeightedMapDataset, weights=weights).seed(42).shuffle()

        elements = [ds[i] for i in range(len(ds))]
        counts = {v: elements.count(v) for v in range(4)}
        assert counts[0] == 4
        assert counts[1] == 2
        assert counts[2] == 1
        assert counts[3] == 1

    def test_repeat_produces_multiple_epochs(self):
        """With repeat, each epoch should have the same element counts."""
        from tsjax.data.sampling import WeightedMapDataset

        parent = grain.MapDataset.source(list(range(3)))
        weights = [2.0, 1.0, 1.0]
        ds = parent.pipe(WeightedMapDataset, weights=weights).seed(0).shuffle().repeat(None)

        epoch_len = 4  # 2 + 1 + 1
        for epoch in range(3):
            start = epoch * epoch_len
            elements = [ds[start + i] for i in range(epoch_len)]
            counts = {v: elements.count(v) for v in range(3)}
            assert counts[0] == 2, f"epoch {epoch}: expected 2 of element 0, got {counts[0]}"
            assert counts[1] == 1
            assert counts[2] == 1

    def test_batch_integration(self):
        """WeightedMapDataset should work with downstream batching."""
        from tsjax.data.sampling import WeightedMapDataset

        parent = grain.MapDataset.source(list(range(4)))
        weights = [2.0, 2.0, 1.0, 1.0]
        # Expanded: 6 elements → 3 batches of 2, no remainder dropped
        ds = (
            parent.pipe(WeightedMapDataset, weights=weights)
            .seed(0)
            .shuffle()
            .batch(2, drop_remainder=True)
        )
        assert len(ds) == 3
        batches = [ds[i] for i in range(len(ds))]
        all_elements = [int(e) for batch in batches for e in batch]
        assert all_elements.count(0) == 2
        assert all_elements.count(1) == 2
        assert all_elements.count(2) == 1
        assert all_elements.count(3) == 1

    def test_length_mismatch_raises(self):
        from tsjax.data.sampling import WeightedMapDataset

        parent = grain.MapDataset.source(list(range(5)))
        with pytest.raises(ValueError, match="weights length"):
            parent.pipe(WeightedMapDataset, weights=[1.0, 2.0])

    def test_negative_weights_raises(self):
        from tsjax.data.sampling import WeightedMapDataset

        parent = grain.MapDataset.source(list(range(3)))
        with pytest.raises(ValueError, match="positive"):
            parent.pipe(WeightedMapDataset, weights=[1.0, -1.0, 1.0])

    def test_zero_weights_raises(self):
        from tsjax.data.sampling import WeightedMapDataset

        parent = grain.MapDataset.source(list(range(3)))
        with pytest.raises(ValueError, match="positive"):
            parent.pipe(WeightedMapDataset, weights=[1.0, 0.0, 1.0])

    def test_fractional_weights_round_correctly(self):
        """Weights like [1.5, 1.0] should round: element 0 gets 2x, element 1 gets 1x."""
        from tsjax.data.sampling import WeightedMapDataset

        parent = grain.MapDataset.source(["a", "b"])
        ds = parent.pipe(WeightedMapDataset, weights=[1.5, 1.0])
        assert len(ds) == 3  # round(1.5)=2 for 'a', 1 for 'b'
        elements = [ds[i] for i in range(len(ds))]
        assert elements.count("a") == 2
        assert elements.count("b") == 1

    def test_dict_elements_work(self):
        """Should work with dict-valued elements (like WindowedSource produces)."""
        from tsjax.data.sampling import WeightedMapDataset

        data = [{"u": np.array([i]), "y": np.array([i * 10])} for i in range(3)]
        parent = grain.MapDataset.source(data)
        ds = parent.pipe(WeightedMapDataset, weights=[2.0, 1.0, 1.0])
        elem = ds[0]
        assert "u" in elem and "y" in elem

    def test_getitems_vectorized(self):
        """_getitems should produce same results as individual __getitem__ calls."""
        from tsjax.data.sampling import WeightedMapDataset

        parent = grain.MapDataset.source(list(range(5)))
        ds = parent.pipe(WeightedMapDataset, weights=[3.0, 1.0, 1.0, 1.0, 1.0])

        indices = [0, 2, 4, 6]
        individual = [ds[i] for i in indices]
        vectorized = ds._getitems(indices)
        assert individual == vectorized


# ---------------------------------------------------------------------------
# uniform_file_weights
# ---------------------------------------------------------------------------


class TestUniformFileWeights:
    @pytest.fixture()
    def unequal_files(self, tmp_path):
        """Create HDF5 files with different sequence lengths."""
        files = []
        for i, length in enumerate([100, 200, 300]):
            path = str(tmp_path / f"file_{i}.hdf5")
            with h5py.File(path, "w") as f:
                f.create_dataset("u", data=np.zeros(length, dtype=np.float32))
                f.create_dataset("y", data=np.zeros(length, dtype=np.float32))
            files.append(path)
        return files

    def test_weights_length_matches_source(self, unequal_files):
        from tsjax import HDF5Store, WindowedSource
        from tsjax.data.sampling import uniform_file_weights

        store = HDF5Store(unequal_files, ["u", "y"])
        source = WindowedSource(store, {"u": ["u"], "y": ["y"]}, win_sz=20, stp_sz=10)
        weights = uniform_file_weights(source)
        assert len(weights) == len(source)

    def test_equal_weight_per_file(self, unequal_files):
        """Total weight per file should be equal regardless of file length."""
        from tsjax import HDF5Store, WindowedSource
        from tsjax.data.sampling import uniform_file_weights

        store = HDF5Store(unequal_files, ["u", "y"])
        source = WindowedSource(store, {"u": ["u"], "y": ["y"]}, win_sz=20, stp_sz=10)
        weights = uniform_file_weights(source)

        cum = source.cum_windows
        file_totals = []
        prev = 0
        for c in cum:
            file_totals.append(weights[prev:c].sum())
            prev = c

        # All files should have equal total weight
        np.testing.assert_allclose(file_totals, file_totals[0], atol=1e-12)

    def test_full_file_mode_returns_ones(self, unequal_files):
        """Full-file mode should return uniform weights."""
        from tsjax import HDF5Store, WindowedSource
        from tsjax.data.sampling import uniform_file_weights

        store = HDF5Store(unequal_files, ["u", "y"])
        source = WindowedSource(store, {"u": ["u"], "y": ["y"]})  # no win_sz
        weights = uniform_file_weights(source)
        np.testing.assert_array_equal(weights, np.ones(3))


# ---------------------------------------------------------------------------
# Pipeline integration
# ---------------------------------------------------------------------------


class TestWeightedPipeline:
    def test_from_sources_with_weights(self, dataset_path):
        """GrainPipeline.from_sources should accept weights."""
        from tsjax import GrainPipeline, HDF5Store, WindowedSource
        from tsjax.data.sampling import uniform_file_weights

        train_files = sorted(str(p) for p in (dataset_path / "train").rglob("*.hdf5"))
        valid_files = sorted(str(p) for p in (dataset_path / "valid").rglob("*.hdf5"))
        test_files = sorted(str(p) for p in (dataset_path / "test").rglob("*.hdf5"))

        def make_src(files, windowed=True):
            store = HDF5Store(files, ["u", "y"])
            specs = {"u": ["u"], "y": ["y"]}
            if windowed:
                return WindowedSource(store, specs, win_sz=20, stp_sz=10)
            return WindowedSource(store, specs)

        train_src = make_src(train_files)
        weights = uniform_file_weights(train_src)

        pl = GrainPipeline.from_sources(
            train_src,
            make_src(valid_files),
            make_src(test_files, windowed=False),
            input_keys=("u",),
            target_keys=("y",),
            bs=4,
            seed=42,
            weights=weights,
        )
        batch = next(iter(pl.train))
        assert batch["u"].shape[0] == 4
        assert batch["u"].shape[1] == 20

    def test_create_grain_dls_with_callable_weights(self, dataset_path):
        """create_grain_dls should accept a callable for weights."""
        from tsjax import create_grain_dls
        from tsjax.data.sampling import uniform_file_weights

        pl = create_grain_dls(
            inputs={"u": ["u"]},
            targets={"y": ["y"]},
            dataset=dataset_path,
            win_sz=20,
            stp_sz=10,
            bs=4,
            weights=uniform_file_weights,
        )
        batch = next(iter(pl.train))
        assert batch["u"].shape == (4, 20, 1)

    def test_create_grain_dls_with_array_weights(self, dataset_path):
        """create_grain_dls should accept an explicit weight array."""
        from tsjax import HDF5Store, WindowedSource, create_grain_dls, discover_split_files

        train_files, _, _ = discover_split_files(dataset_path)
        store = HDF5Store(train_files, ["u", "y"])
        source = WindowedSource(store, {"u": ["u"], "y": ["y"]}, win_sz=20, stp_sz=10)
        n = len(source)

        pl = create_grain_dls(
            inputs={"u": ["u"]},
            targets={"y": ["y"]},
            dataset=dataset_path,
            win_sz=20,
            stp_sz=10,
            bs=4,
            weights=np.ones(n),
        )
        batch = next(iter(pl.train))
        assert batch["u"].shape == (4, 20, 1)

    def test_weighted_pipeline_changes_n_train_batches(self, dataset_path):
        """Weights that expand the dataset should increase n_train_batches."""
        from tsjax import create_grain_dls
        from tsjax.data.sampling import uniform_file_weights

        pl_uniform = create_grain_dls(
            inputs={"u": ["u"]},
            targets={"y": ["y"]},
            dataset=dataset_path,
            win_sz=20,
            stp_sz=10,
            bs=4,
        )

        pl_weighted = create_grain_dls(
            inputs={"u": ["u"]},
            targets={"y": ["y"]},
            dataset=dataset_path,
            win_sz=20,
            stp_sz=10,
            bs=4,
            weights=uniform_file_weights,
        )

        # With file balancing, n_train_batches may differ from uniform
        # (it should be >= uniform since weights expand the index space)
        assert pl_weighted.n_train_batches >= pl_uniform.n_train_batches
