"""Tests for the per-sample transform pipeline (Phase 3)."""

from __future__ import annotations

import h5py
import numpy as np
import pytest


@pytest.fixture()
def transform_dataset(tmp_path):
    """Create a minimal dataset with train/valid/test splits."""
    rng = np.random.default_rng(42)
    for split in ("train", "valid", "test"):
        d = tmp_path / split
        d.mkdir()
        for i in range(3):
            with h5py.File(str(d / f"file_{i}.hdf5"), "w") as f:
                f.create_dataset("u", data=rng.standard_normal(100).astype(np.float32))
                f.create_dataset("y", data=rng.standard_normal(100).astype(np.float32))
    return tmp_path


# ---------------------------------------------------------------------------
# _apply_transforms unit tests
# ---------------------------------------------------------------------------


class TestApplyTransforms:
    def test_identity_preserves_data(self):
        from tsjax.data.item_transforms import _apply_transforms

        sample = {"u": np.array([1.0, 2.0, 3.0]), "y": np.array([4.0, 5.0, 6.0])}
        fn = _apply_transforms({"u": lambda x: x})
        result = fn(sample)
        np.testing.assert_array_equal(result["u"], sample["u"])
        np.testing.assert_array_equal(result["y"], sample["y"])

    def test_transforms_only_specified_keys(self):
        from tsjax.data.item_transforms import _apply_transforms

        sample = {"u": np.array([1.0, 2.0]), "y": np.array([3.0, 4.0])}
        fn = _apply_transforms({"u": lambda x: x * 2})
        result = fn(sample)
        np.testing.assert_array_equal(result["u"], np.array([2.0, 4.0]))
        np.testing.assert_array_equal(result["y"], np.array([3.0, 4.0]))

    def test_multiple_keys_transformed(self):
        from tsjax.data.item_transforms import _apply_transforms

        sample = {"a": np.array([1.0]), "b": np.array([2.0]), "c": np.array([3.0])}
        fn = _apply_transforms({"a": lambda x: x + 10, "b": lambda x: x * 3})
        result = fn(sample)
        np.testing.assert_array_equal(result["a"], np.array([11.0]))
        np.testing.assert_array_equal(result["b"], np.array([6.0]))
        np.testing.assert_array_equal(result["c"], np.array([3.0]))


# ---------------------------------------------------------------------------
# Pipeline integration tests
# ---------------------------------------------------------------------------


class TestPipelineWithTransforms:
    def test_identity_transform_unchanged(self, transform_dataset):
        """Identity transform should not alter batch data."""
        from tsjax.data.pipeline import create_grain_dls

        pl_no_xform = create_grain_dls(
            inputs={"u": ["u"]},
            targets={"y": ["y"]},
            dataset=transform_dataset,
            win_sz=20,
            stp_sz=20,
            bs=2,
            seed=0,
        )
        pl_identity = create_grain_dls(
            inputs={"u": ["u"]},
            targets={"y": ["y"]},
            dataset=transform_dataset,
            win_sz=20,
            stp_sz=20,
            bs=2,
            seed=0,
            transforms={"u": lambda x: x},
        )

        batch_no = pl_no_xform.train[0]
        batch_id = pl_identity.train[0]
        np.testing.assert_array_equal(batch_no["u"], batch_id["u"])
        np.testing.assert_array_equal(batch_no["y"], batch_id["y"])

    def test_scaling_transform_applied(self, transform_dataset):
        """A 2x scaling transform should double the values."""
        from tsjax.data.pipeline import create_grain_dls

        pl_base = create_grain_dls(
            inputs={"u": ["u"]},
            targets={"y": ["y"]},
            dataset=transform_dataset,
            win_sz=20,
            stp_sz=20,
            bs=2,
            seed=0,
        )
        pl_scaled = create_grain_dls(
            inputs={"u": ["u"]},
            targets={"y": ["y"]},
            dataset=transform_dataset,
            win_sz=20,
            stp_sz=20,
            bs=2,
            seed=0,
            transforms={"u": lambda x: x * 2},
        )

        batch_base = pl_base.train[0]
        batch_scaled = pl_scaled.train[0]
        np.testing.assert_allclose(batch_scaled["u"], batch_base["u"] * 2, atol=1e-6)
        np.testing.assert_array_equal(batch_scaled["y"], batch_base["y"])

    def test_stats_computed_on_transformed_data(self, transform_dataset):
        """Stats should reflect transformed data: 2x scale -> 2x mean, 2x std."""
        from tsjax.data.pipeline import create_grain_dls

        pl_base = create_grain_dls(
            inputs={"u": ["u"]},
            targets={"y": ["y"]},
            dataset=transform_dataset,
            win_sz=20,
            stp_sz=20,
            bs=2,
        )
        pl_scaled = create_grain_dls(
            inputs={"u": ["u"]},
            targets={"y": ["y"]},
            dataset=transform_dataset,
            win_sz=20,
            stp_sz=20,
            bs=2,
            transforms={"u": lambda x: x * 2},
        )

        np.testing.assert_allclose(
            pl_scaled.stats["u"].mean, pl_base.stats["u"].mean * 2, atol=1e-4
        )
        np.testing.assert_allclose(
            pl_scaled.stats["u"].std, pl_base.stats["u"].std * 2, atol=1e-4
        )
        # y stats should be unchanged
        np.testing.assert_array_equal(pl_scaled.stats["y"].mean, pl_base.stats["y"].mean)
        np.testing.assert_array_equal(pl_scaled.stats["y"].std, pl_base.stats["y"].std)

    def test_shape_changing_transform(self, transform_dataset):
        """A transform that reduces (win_sz, 1) -> (1,)."""
        from functools import partial

        from tsjax.data.pipeline import create_grain_dls

        pl = create_grain_dls(
            inputs={"u": ["u"]},
            targets={"y": ["y"]},
            dataset=transform_dataset,
            win_sz=20,
            stp_sz=20,
            bs=2,
            transforms={"u": partial(np.mean, axis=0)},
        )

        batch = pl.train[0]
        assert batch["u"].shape == (2, 1)  # reduced to scalar per channel
        assert batch["y"].shape == (2, 20, 1)  # unchanged

    def test_invalid_transform_key_raises(self, transform_dataset):
        from tsjax.data.pipeline import create_grain_dls

        with pytest.raises(ValueError, match="Transform keys"):
            create_grain_dls(
                inputs={"u": ["u"]},
                targets={"y": ["y"]},
                dataset=transform_dataset,
                win_sz=20,
                stp_sz=20,
                bs=2,
                transforms={"nonexistent": lambda x: x},
            )

    def test_transforms_applied_to_all_splits(self, transform_dataset):
        """Transforms should apply to train, valid, and test splits."""
        from tsjax.data.pipeline import create_grain_dls

        pl = create_grain_dls(
            inputs={"u": ["u"]},
            targets={"y": ["y"]},
            dataset=transform_dataset,
            win_sz=20,
            stp_sz=20,
            bs=2,
            transforms={"u": lambda x: x * 3},
        )
        pl_base = create_grain_dls(
            inputs={"u": ["u"]},
            targets={"y": ["y"]},
            dataset=transform_dataset,
            win_sz=20,
            stp_sz=20,
            bs=2,
        )

        # Check valid split
        v_batch = pl.valid[0]
        v_base = pl_base.valid[0]
        np.testing.assert_allclose(v_batch["u"], v_base["u"] * 3, atol=1e-6)

        # Check test split (full sequence, batch=1)
        t_batch = pl.test[0]
        t_base = pl_base.test[0]
        np.testing.assert_allclose(t_batch["u"], t_base["u"] * 3, atol=1e-6)


# ---------------------------------------------------------------------------
# compute_stats_with_transform unit tests
# ---------------------------------------------------------------------------


class TestComputeStatsWithTransform:
    def test_identity_stats_close_to_raw(self, transform_dataset):
        """Stats with identity transform should closely match raw stats."""
        from tsjax.data.pipeline import create_grain_dls
        from tsjax.data.stats import compute_stats_with_transform

        pl = create_grain_dls(
            inputs={"u": ["u"]},
            targets={"y": ["y"]},
            dataset=transform_dataset,
            win_sz=20,
            stp_sz=20,
            bs=2,
        )

        xform_stats = compute_stats_with_transform(pl.train_source, "u", lambda x: x)

        # With non-overlapping windows covering all samples, stats should be very close
        np.testing.assert_allclose(xform_stats.mean, pl.stats["u"].mean, atol=1e-3)
        np.testing.assert_allclose(xform_stats.std, pl.stats["u"].std, atol=1e-2)

    def test_scalar_output_transform(self, transform_dataset):
        """Stats on a scalar-output transform should have correct shape."""
        from functools import partial

        from tsjax.data.pipeline import create_grain_dls
        from tsjax.data.stats import compute_stats_with_transform

        pl = create_grain_dls(
            inputs={"u": ["u"]},
            targets={"y": ["y"]},
            dataset=transform_dataset,
            win_sz=20,
            stp_sz=20,
            bs=2,
        )

        stats = compute_stats_with_transform(
            pl.train_source, "u", partial(np.mean, axis=0)
        )
        assert stats.mean.shape == (1,)
        assert stats.std.shape == (1,)
        assert stats.mean.dtype == np.float32

    def test_dtype_always_float32(self, transform_dataset):
        from tsjax.data.pipeline import create_grain_dls
        from tsjax.data.stats import compute_stats_with_transform

        pl = create_grain_dls(
            inputs={"u": ["u"]},
            targets={"y": ["y"]},
            dataset=transform_dataset,
            win_sz=20,
            stp_sz=20,
            bs=2,
        )

        stats = compute_stats_with_transform(pl.train_source, "u", lambda x: x * 100)
        assert stats.mean.dtype == np.float32
        assert stats.std.dtype == np.float32
