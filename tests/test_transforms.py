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

        batch_no = next(iter(pl_no_xform.train_loader(0)))
        batch_id = next(iter(pl_identity.train_loader(0)))
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

        batch_base = next(iter(pl_base.train_loader(0)))
        batch_scaled = next(iter(pl_scaled.train_loader(0)))
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
        np.testing.assert_allclose(pl_scaled.stats["u"].std, pl_base.stats["u"].std * 2, atol=1e-4)
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

        batch = next(iter(pl.train_loader(0)))
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
        v_batch = next(iter(pl.valid))
        v_base = next(iter(pl_base.valid))
        np.testing.assert_allclose(v_batch["u"], v_base["u"] * 3, atol=1e-6)

        # Check test split (full sequence, batch=1)
        t_batch = next(iter(pl.test))
        t_base = next(iter(pl_base.test))
        np.testing.assert_allclose(t_batch["u"], t_base["u"] * 3, atol=1e-6)


# ---------------------------------------------------------------------------
# compute_stats_with_transform unit tests
# ---------------------------------------------------------------------------


class TestComputeStatsWithTransform:
    def test_identity_stats_close_to_raw(self, transform_dataset):
        """Stats with identity transform should closely match raw stats."""
        from tsjax.data.pipeline import create_grain_dls
        from tsjax.data.stats import compute_stats

        pl = create_grain_dls(
            inputs={"u": ["u"]},
            targets={"y": ["y"]},
            dataset=transform_dataset,
            win_sz=20,
            stp_sz=20,
            bs=2,
        )

        xform_stats = compute_stats(pl.train_source, "u", transform=lambda x: x)

        # With non-overlapping windows covering all samples, stats should be very close
        np.testing.assert_allclose(xform_stats.mean, pl.stats["u"].mean, atol=1e-3)
        np.testing.assert_allclose(xform_stats.std, pl.stats["u"].std, atol=1e-2)

    def test_scalar_output_transform(self, transform_dataset):
        """Stats on a scalar-output transform should have correct shape."""
        from functools import partial

        from tsjax.data.pipeline import create_grain_dls
        from tsjax.data.stats import compute_stats

        pl = create_grain_dls(
            inputs={"u": ["u"]},
            targets={"y": ["y"]},
            dataset=transform_dataset,
            win_sz=20,
            stp_sz=20,
            bs=2,
        )

        stats = compute_stats(pl.train_source, "u", transform=partial(np.mean, axis=0))
        assert stats.mean.shape == (1,)
        assert stats.std.shape == (1,)
        assert stats.mean.dtype == np.float32

    def test_dtype_always_float32(self, transform_dataset):
        from tsjax.data.pipeline import create_grain_dls
        from tsjax.data.stats import compute_stats

        pl = create_grain_dls(
            inputs={"u": ["u"]},
            targets={"y": ["y"]},
            dataset=transform_dataset,
            win_sz=20,
            stp_sz=20,
            bs=2,
        )

        stats = compute_stats(pl.train_source, "u", transform=lambda x: x * 100)
        assert stats.mean.dtype == np.float32
        assert stats.std.dtype == np.float32


# ---------------------------------------------------------------------------
# seq_slice unit tests
# ---------------------------------------------------------------------------


class TestSeqSlice:
    def test_left_slice(self):
        from tsjax.data.item_transforms import seq_slice

        x = np.arange(10).reshape(10, 1).astype(np.float32)
        result = seq_slice(l_slc=3)(x)
        np.testing.assert_array_equal(result, x[3:])

    def test_right_slice(self):
        from tsjax.data.item_transforms import seq_slice

        x = np.arange(10).reshape(10, 1).astype(np.float32)
        result = seq_slice(r_slc=-2)(x)
        np.testing.assert_array_equal(result, x[:-2])

    def test_both_slices(self):
        from tsjax.data.item_transforms import seq_slice

        x = np.arange(10).reshape(10, 1).astype(np.float32)
        result = seq_slice(l_slc=2, r_slc=-1)(x)
        np.testing.assert_array_equal(result, x[2:-1])

    def test_none_slices_identity(self):
        from tsjax.data.item_transforms import seq_slice

        x = np.arange(10).reshape(10, 1).astype(np.float32)
        result = seq_slice()(x)
        np.testing.assert_array_equal(result, x)


# ---------------------------------------------------------------------------
# _apply_augmentations unit tests
# ---------------------------------------------------------------------------


class TestApplyAugmentations:
    def test_augments_specified_keys(self):
        from tsjax.data.item_transforms import _apply_augmentations

        sample = {"u": np.ones((5, 1), dtype=np.float32), "y": np.ones((5, 1), dtype=np.float32)}
        fn = _apply_augmentations({"u": lambda x, rng: x + 1.0})
        rng = np.random.default_rng(42)
        result = fn(sample, rng)
        np.testing.assert_array_equal(result["u"], np.full((5, 1), 2.0))
        np.testing.assert_array_equal(result["y"], np.ones((5, 1)))

    def test_passes_rng_to_augmentation(self):
        from tsjax.data.item_transforms import _apply_augmentations

        sample = {"u": np.zeros((5, 1), dtype=np.float32)}

        def _noisy(x, rng):
            return x + rng.normal(0, 1, x.shape).astype(x.dtype)

        fn = _apply_augmentations({"u": _noisy})
        rng = np.random.default_rng(42)
        result = fn(sample, rng)
        assert not np.allclose(result["u"], 0.0)


# ---------------------------------------------------------------------------
# Augmentation factory unit tests
# ---------------------------------------------------------------------------


class TestNoiseInjection:
    def test_adds_noise(self):
        from tsjax.data.item_transforms import noise_injection

        aug = noise_injection(std=1.0, mean=0.0)
        x = np.zeros((100, 2), dtype=np.float32)
        rng = np.random.default_rng(42)
        result = aug(x, rng)
        assert not np.allclose(result, 0.0)
        assert result.shape == x.shape
        assert result.dtype == x.dtype

    def test_per_channel_std(self):
        from tsjax.data.item_transforms import noise_injection

        aug = noise_injection(std=np.array([0.01, 100.0]))
        x = np.zeros((1000, 2), dtype=np.float32)
        rng = np.random.default_rng(42)
        result = aug(x, rng)
        assert np.std(result[:, 1]) > np.std(result[:, 0]) * 10

    def test_probability_zero_no_change(self):
        from tsjax.data.item_transforms import noise_injection

        aug = noise_injection(std=1.0, p=0.0)
        x = np.ones((10, 1), dtype=np.float32)
        rng = np.random.default_rng(42)
        result = aug(x, rng)
        np.testing.assert_array_equal(result, x)

    def test_probability_one_always_applies(self):
        from tsjax.data.item_transforms import noise_injection

        aug = noise_injection(std=1.0, p=1.0)
        x = np.zeros((10, 1), dtype=np.float32)
        rng = np.random.default_rng(42)
        result = aug(x, rng)
        assert not np.allclose(result, 0.0)


class TestVaryingNoise:
    def test_adds_noise(self):
        from tsjax.data.item_transforms import varying_noise

        aug = varying_noise(std_std=1.0)
        x = np.zeros((100, 1), dtype=np.float32)
        rng = np.random.default_rng(42)
        result = aug(x, rng)
        assert not np.allclose(result, 0.0)

    def test_probability_zero(self):
        from tsjax.data.item_transforms import varying_noise

        aug = varying_noise(std_std=1.0, p=0.0)
        x = np.ones((10, 1), dtype=np.float32)
        rng = np.random.default_rng(42)
        result = aug(x, rng)
        np.testing.assert_array_equal(result, x)


class TestGroupedNoise:
    def test_group_structure(self):
        from tsjax.data.item_transforms import grouped_noise

        aug = grouped_noise(
            std_std=np.array([0.01, 100.0]),
            std_idx=np.array([0, 0, 1, 1]),
        )
        x = np.zeros((1000, 4), dtype=np.float32)
        rng = np.random.default_rng(42)
        result = aug(x, rng)
        group0_std = np.std(result[:, :2])
        group1_std = np.std(result[:, 2:])
        assert group1_std > group0_std * 10

    def test_probability_zero(self):
        from tsjax.data.item_transforms import grouped_noise

        aug = grouped_noise(std_std=np.array([1.0]), std_idx=np.array([0]), p=0.0)
        x = np.ones((10, 1), dtype=np.float32)
        rng = np.random.default_rng(42)
        result = aug(x, rng)
        np.testing.assert_array_equal(result, x)


class TestBiasInjection:
    def test_constant_offset_across_time(self):
        from tsjax.data.item_transforms import bias_injection

        aug = bias_injection(std=1.0, mean=0.0)
        x = np.zeros((50, 2), dtype=np.float32)
        rng = np.random.default_rng(42)
        result = aug(x, rng)
        for ch in range(2):
            np.testing.assert_array_equal(result[:, ch], result[0, ch])

    def test_probability_zero(self):
        from tsjax.data.item_transforms import bias_injection

        aug = bias_injection(std=1.0, p=0.0)
        x = np.ones((10, 1), dtype=np.float32)
        rng = np.random.default_rng(42)
        result = aug(x, rng)
        np.testing.assert_array_equal(result, x)


class TestChainAugmentations:
    def test_chains_two_augmentations(self):
        from tsjax.data.item_transforms import bias_injection, chain_augmentations, noise_injection

        aug = chain_augmentations(noise_injection(std=0.01), bias_injection(std=0.01))
        x = np.zeros((20, 1), dtype=np.float32)
        rng = np.random.default_rng(42)
        result = aug(x, rng)
        assert not np.allclose(result, 0.0)
        # Values should not be identical across time (noise component)
        assert not np.allclose(result, result[0])

    def test_empty_chain_is_identity(self):
        from tsjax.data.item_transforms import chain_augmentations

        aug = chain_augmentations()
        x = np.ones((5, 1), dtype=np.float32)
        rng = np.random.default_rng(42)
        result = aug(x, rng)
        np.testing.assert_array_equal(result, x)


# ---------------------------------------------------------------------------
# Pipeline integration tests for augmentations
# ---------------------------------------------------------------------------


class TestPipelineWithAugmentations:
    def test_augmentation_only_on_train(self, transform_dataset):
        """Augmentations should only apply to training data."""
        from tsjax.data.pipeline import create_grain_dls

        pl_aug = create_grain_dls(
            inputs={"u": ["u"]},
            targets={"y": ["y"]},
            dataset=transform_dataset,
            win_sz=20,
            stp_sz=20,
            bs=2,
            seed=0,
            augmentations={"u": lambda x, rng: x + 100.0},
        )
        pl_base = create_grain_dls(
            inputs={"u": ["u"]},
            targets={"y": ["y"]},
            dataset=transform_dataset,
            win_sz=20,
            stp_sz=20,
            bs=2,
            seed=0,
        )
        np.testing.assert_array_equal(next(iter(pl_aug.valid))["u"], next(iter(pl_base.valid))["u"])
        np.testing.assert_array_equal(next(iter(pl_aug.test))["u"], next(iter(pl_base.test))["u"])

    def test_augmentation_does_not_affect_stats(self, transform_dataset):
        """Stats should be identical with or without augmentations."""
        from tsjax.data.pipeline import create_grain_dls

        pl_aug = create_grain_dls(
            inputs={"u": ["u"]},
            targets={"y": ["y"]},
            dataset=transform_dataset,
            win_sz=20,
            stp_sz=20,
            bs=2,
            augmentations={"u": lambda x, rng: x + 100.0},
        )
        pl_base = create_grain_dls(
            inputs={"u": ["u"]},
            targets={"y": ["y"]},
            dataset=transform_dataset,
            win_sz=20,
            stp_sz=20,
            bs=2,
        )
        np.testing.assert_array_equal(pl_aug.stats["u"].mean, pl_base.stats["u"].mean)
        np.testing.assert_array_equal(pl_aug.stats["u"].std, pl_base.stats["u"].std)

    def test_augmentation_modifies_train_data(self, transform_dataset):
        """Augmented training data should differ from non-augmented."""
        from tsjax.data.pipeline import create_grain_dls

        pl_aug = create_grain_dls(
            inputs={"u": ["u"]},
            targets={"y": ["y"]},
            dataset=transform_dataset,
            win_sz=20,
            stp_sz=20,
            bs=2,
            seed=0,
            augmentations={"u": lambda x, rng: x + 100.0},
        )
        pl_base = create_grain_dls(
            inputs={"u": ["u"]},
            targets={"y": ["y"]},
            dataset=transform_dataset,
            win_sz=20,
            stp_sz=20,
            bs=2,
            seed=0,
        )
        diff = np.abs(
            next(iter(pl_aug.train_loader(0)))["u"] - next(iter(pl_base.train_loader(0)))["u"]
        )
        assert np.all(diff > 50)

    def test_invalid_augmentation_key_raises(self, transform_dataset):
        from tsjax.data.pipeline import create_grain_dls

        with pytest.raises(ValueError, match="Augmentation keys"):
            create_grain_dls(
                inputs={"u": ["u"]},
                targets={"y": ["y"]},
                dataset=transform_dataset,
                win_sz=20,
                stp_sz=20,
                bs=2,
                augmentations={"nonexistent": lambda x, rng: x},
            )

    def test_transforms_and_augmentations_compose(self, transform_dataset):
        """Transforms and augmentations can be used together."""
        from tsjax.data.pipeline import create_grain_dls

        pl = create_grain_dls(
            inputs={"u": ["u"]},
            targets={"y": ["y"]},
            dataset=transform_dataset,
            win_sz=20,
            stp_sz=20,
            bs=2,
            seed=0,
            transforms={"u": lambda x: x * 2},
            augmentations={"u": lambda x, rng: x + 100.0},
        )
        pl_xform_only = create_grain_dls(
            inputs={"u": ["u"]},
            targets={"y": ["y"]},
            dataset=transform_dataset,
            win_sz=20,
            stp_sz=20,
            bs=2,
            seed=0,
            transforms={"u": lambda x: x * 2},
        )
        # Stats should match (augmentations don't affect stats)
        np.testing.assert_array_equal(pl.stats["u"].mean, pl_xform_only.stats["u"].mean)
        # Valid data should match (augmentations are train-only)
        np.testing.assert_array_equal(
            next(iter(pl.valid))["u"], next(iter(pl_xform_only.valid))["u"]
        )
