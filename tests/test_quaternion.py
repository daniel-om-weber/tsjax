"""Tests for quaternion algebra, losses, metrics, augmentation, and SLERP."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest

from tsjax.quaternion import (
    abs_inclination,
    abs_rel_angle,
    inclination_angle,
    inclination_loss,
    inclination_loss_abs,
    mean_inclination_deg,
    mean_rel_angle_deg,
    ms_inclination,
    ms_rel_angle,
    nan_safe,
    pitch_angle,
    quat_conjugate,
    quat_diff,
    quat_multiply,
    quat_normalize,
    quat_relative,
    relative_angle,
    rms_inclination,
    rms_inclination_deg,
    rms_pitch_deg,
    rms_rel_angle_deg,
    rms_roll_deg,
    roll_angle,
    rot_vec,
    smooth_inclination,
)

IDENTITY = jnp.array([1.0, 0.0, 0.0, 0.0])
# 90-deg rotation around z-axis: cos(45deg), 0, 0, sin(45deg)
ROT_Z90 = jnp.array([np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)])
# 90-deg rotation around x-axis (causes inclination change, unlike z-rotation)
ROT_X90 = jnp.array([np.cos(np.pi / 4), np.sin(np.pi / 4), 0.0, 0.0])
DUMMY_STATS = jnp.zeros(4), jnp.ones(4)  # y_mean, y_std (ignored by quat losses)


# ---------------------------------------------------------------------------
# Core algebra
# ---------------------------------------------------------------------------


class TestQuatMultiply:
    def test_identity(self):
        q = jnp.array([0.5, 0.5, 0.5, 0.5])
        result = quat_multiply(q, IDENTITY)
        npt.assert_allclose(result, q, atol=1e-6)

    def test_inverse(self):
        q = quat_normalize(jnp.array([1.0, 2.0, 3.0, 4.0]))
        result = quat_multiply(q, quat_conjugate(q))
        npt.assert_allclose(result, IDENTITY, atol=1e-6)

    def test_known_rotation(self):
        # 90-deg around z twice = 180-deg around z
        result = quat_multiply(ROT_Z90, ROT_Z90)
        expected = jnp.array([0.0, 0.0, 0.0, 1.0])  # cos(90), 0, 0, sin(90)
        npt.assert_allclose(result, expected, atol=1e-6)

    def test_batch(self):
        q1 = jnp.stack([IDENTITY, ROT_Z90])
        q2 = jnp.stack([ROT_Z90, IDENTITY])
        result = quat_multiply(q1, q2)
        assert result.shape == (2, 4)
        npt.assert_allclose(result[0], ROT_Z90, atol=1e-6)
        npt.assert_allclose(result[1], ROT_Z90, atol=1e-6)

    def test_sequence_batch(self):
        q = jnp.ones((2, 5, 4)) * 0.5  # (batch, seq, 4)
        result = quat_multiply(q, q)
        assert result.shape == (2, 5, 4)


class TestQuatConjugate:
    def test_sign_flip(self):
        q = jnp.array([1.0, 2.0, 3.0, 4.0])
        result = quat_conjugate(q)
        npt.assert_allclose(result, jnp.array([1.0, -2.0, -3.0, -4.0]))

    def test_identity_is_self(self):
        result = quat_conjugate(IDENTITY)
        npt.assert_allclose(result, IDENTITY)


class TestQuatNormalize:
    def test_unit_norm(self):
        q = jnp.array([1.0, 2.0, 3.0, 4.0])
        result = quat_normalize(q)
        npt.assert_allclose(jnp.linalg.norm(result), 1.0, atol=1e-6)

    def test_already_normalized(self):
        result = quat_normalize(IDENTITY)
        npt.assert_allclose(result, IDENTITY, atol=1e-6)


class TestQuatRelative:
    def test_same_gives_identity(self):
        q = quat_normalize(jnp.array([1.0, 2.0, 3.0, 4.0]))
        result = quat_relative(q, q)
        npt.assert_allclose(result, IDENTITY, atol=1e-6)

    def test_inverse(self):
        result = quat_relative(ROT_Z90, IDENTITY)
        npt.assert_allclose(result, ROT_Z90, atol=1e-6)


class TestQuatDiff:
    def test_matches_relative_without_normalize(self):
        q1 = quat_normalize(jnp.array([1.0, 2.0, 3.0, 4.0]))
        q2 = quat_normalize(jnp.array([4.0, 3.0, 2.0, 1.0]))
        result = quat_diff(q1, q2, normalize=False)
        expected = quat_relative(q1, q2)
        npt.assert_allclose(result, expected, atol=1e-6)


class TestRotVec:
    def test_identity_rotation(self):
        v = jnp.array([1.0, 0.0, 0.0])
        result = rot_vec(v, IDENTITY)
        npt.assert_allclose(result, v, atol=1e-6)

    def test_90deg_z(self):
        # rot_vec uses conj(q)*v*q convention (inverse rotation), matching TSFast
        v = jnp.array([1.0, 0.0, 0.0])
        result = rot_vec(v, ROT_Z90)
        expected = jnp.array([0.0, -1.0, 0.0])
        npt.assert_allclose(result, expected, atol=1e-6)

    def test_batch(self):
        v = jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        result = rot_vec(v, ROT_Z90)
        assert result.shape == (2, 3)


# ---------------------------------------------------------------------------
# Angle extraction
# ---------------------------------------------------------------------------


class TestInclinationAngle:
    def test_zero_for_identical(self):
        q = quat_normalize(jnp.array([1.0, 2.0, 3.0, 4.0]))
        result = inclination_angle(q, q)
        npt.assert_allclose(result, 0.0, atol=1e-3)

    def test_positive_for_different(self):
        # Use x-rotation which changes inclination (unlike z-rotation)
        result = inclination_angle(IDENTITY, ROT_X90)
        assert float(result) > 0

    def test_batch(self):
        q1 = jnp.stack([IDENTITY, ROT_X90])
        q2 = jnp.stack([ROT_X90, IDENTITY])
        result = inclination_angle(q1, q2)
        assert result.shape == (2,)


class TestRelativeAngle:
    def test_zero_for_identical(self):
        result = relative_angle(ROT_Z90, ROT_Z90)
        npt.assert_allclose(result, 0.0, atol=1e-3)

    def test_known_angle(self):
        # Relative angle between identity and 90-deg-z should be ~pi/2
        result = relative_angle(IDENTITY, ROT_Z90)
        npt.assert_allclose(result, jnp.pi / 2, atol=1e-3)


class TestRollPitchAngle:
    def test_zero_for_identical(self):
        npt.assert_allclose(roll_angle(IDENTITY, IDENTITY), 0.0, atol=1e-5)
        npt.assert_allclose(pitch_angle(IDENTITY, IDENTITY), 0.0, atol=1e-5)


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------


class TestLossFunctions:
    """All losses take (pred, target, y_mean, y_std) and return a scalar."""

    @pytest.fixture
    def identical_quats(self):
        q = jnp.tile(quat_normalize(jnp.array([1.0, 0.5, 0.3, 0.1])), (4, 10, 1))
        return q, q

    # Use x-rotation for "different" quaternions — ensures inclination change
    @pytest.fixture
    def different_quats(self):
        q1 = jnp.tile(IDENTITY, (4, 10, 1))
        q2 = jnp.tile(ROT_X90, (4, 10, 1))
        return q1, q2

    LOSS_FNS = [
        abs_inclination,
        ms_inclination,
        rms_inclination,
        smooth_inclination,
        inclination_loss,
        inclination_loss_abs,
        ms_rel_angle,
        abs_rel_angle,
    ]

    @pytest.mark.parametrize("fn", LOSS_FNS, ids=lambda f: f.__name__)
    def test_zero_for_identical(self, fn, identical_quats):
        q, _ = identical_quats
        result = fn(q, q, *DUMMY_STATS)
        npt.assert_allclose(result, 0.0, atol=1e-3)

    @pytest.mark.parametrize("fn", LOSS_FNS, ids=lambda f: f.__name__)
    def test_positive_for_different(self, fn, different_quats):
        q1, q2 = different_quats
        result = fn(q1, q2, *DUMMY_STATS)
        assert float(result) > 0

    @pytest.mark.parametrize("fn", LOSS_FNS, ids=lambda f: f.__name__)
    def test_ignores_stats(self, fn, different_quats):
        q1, q2 = different_quats
        r1 = fn(q1, q2, jnp.zeros(4), jnp.ones(4))
        r2 = fn(q1, q2, jnp.ones(4) * 100, jnp.ones(4) * 50)
        npt.assert_allclose(r1, r2, atol=1e-6)

    @pytest.mark.parametrize("fn", LOSS_FNS, ids=lambda f: f.__name__)
    def test_scalar_output(self, fn, different_quats):
        q1, q2 = different_quats
        result = fn(q1, q2, *DUMMY_STATS)
        assert result.shape == ()


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


class TestMetrics:
    METRIC_FNS = [
        rms_inclination_deg,
        mean_inclination_deg,
        rms_pitch_deg,
        rms_roll_deg,
        rms_rel_angle_deg,
        mean_rel_angle_deg,
    ]

    @pytest.mark.parametrize("fn", METRIC_FNS, ids=lambda f: f.__name__)
    def test_zero_for_identical(self, fn):
        q = jnp.tile(IDENTITY, (4, 10, 1))
        result = fn(q, q, *DUMMY_STATS)
        npt.assert_allclose(result, 0.0, atol=0.1)

    @pytest.mark.parametrize("fn", METRIC_FNS, ids=lambda f: f.__name__)
    def test_plausible_degree_range(self, fn):
        # Asymmetric rotation so all Euler angle components are non-zero
        q_gen = quat_normalize(jnp.array([0.9, 0.3, 0.2, 0.1]))
        q1 = jnp.tile(IDENTITY, (4, 10, 1))
        q2 = jnp.tile(q_gen, (4, 10, 1))
        result = float(fn(q1, q2, *DUMMY_STATS))
        assert 0 < abs(result) < 360

    def test_rms_inclination_deg_known(self):
        q1 = jnp.tile(IDENTITY, (4, 10, 1))
        q2 = jnp.tile(ROT_X90, (4, 10, 1))
        rad_val = float(rms_inclination(q1, q2, *DUMMY_STATS))
        deg_val = float(rms_inclination_deg(q1, q2, *DUMMY_STATS))
        npt.assert_allclose(deg_val, np.degrees(rad_val), atol=1e-3)


# ---------------------------------------------------------------------------
# NaN-safe wrapper
# ---------------------------------------------------------------------------


class TestNanSafe:
    def test_no_nan_unchanged(self):
        q1 = jnp.tile(IDENTITY, (4, 10, 1))
        q2 = jnp.tile(ROT_X90, (4, 10, 1))
        raw = abs_inclination(q1, q2, *DUMMY_STATS)
        safe_fn = nan_safe(abs_inclination)
        result = safe_fn(q1, q2, *DUMMY_STATS)
        npt.assert_allclose(result, raw, atol=1e-6)

    def test_all_nan_near_zero(self):
        q1 = jnp.tile(IDENTITY, (4, 10, 1))
        q2 = jnp.full((4, 10, 4), jnp.nan)
        safe_fn = nan_safe(abs_inclination)
        result = safe_fn(q1, q2, *DUMMY_STATS)
        # Masked elements become identity-vs-identity → zero error
        assert float(result) < 0.1

    def test_partial_nan_corrects_mean(self):
        # Half NaN, half real — mean should reflect only real elements
        q1 = jnp.tile(IDENTITY, (4, 1, 1))
        q2_real = jnp.tile(ROT_X90, (2, 1, 1))
        q2_nan = jnp.full((2, 1, 4), jnp.nan)
        q2 = jnp.concatenate([q2_real, q2_nan], axis=0)

        safe_fn = nan_safe(abs_inclination)
        result = float(safe_fn(q1, q2, *DUMMY_STATS))

        # The full (no-nan) value on just the valid portion
        full = float(abs_inclination(q1[:2], q2_real, *DUMMY_STATS))
        npt.assert_allclose(result, full, atol=1e-2)


# ---------------------------------------------------------------------------
# NumPy helpers (_quat_np)
# ---------------------------------------------------------------------------


class TestQuatNp:
    def test_multiply_matches_jax(self):
        from tsjax.quaternion._np import multiply as np_mul

        q1 = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)
        q2 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        np_result = np_mul(q1, q2)
        jax_result = np.asarray(quat_multiply(jnp.array(q1), jnp.array(q2)))
        npt.assert_allclose(np_result, jax_result, atol=1e-6)

    def test_rot_vec_matches_jax(self):
        from tsjax.quaternion._np import rot_vec as np_rot

        v = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        q = np.array(ROT_Z90, dtype=np.float32)
        np_result = np_rot(v, q)
        jax_result = np.asarray(rot_vec(jnp.array(v), jnp.array(q)))
        npt.assert_allclose(np_result, jax_result, atol=1e-5)

    def test_relative_matches_jax(self):
        from tsjax.quaternion._np import relative as np_rel

        q1 = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)
        q2 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        np_result = np_rel(q1, q2)
        jax_result = np.asarray(quat_relative(jnp.array(q1), jnp.array(q2)))
        npt.assert_allclose(np_result, jax_result, atol=1e-6)

    def test_rand_quat_unit_norm(self):
        from tsjax.quaternion._np import rand_quat

        rng = np.random.default_rng(42)
        for _ in range(10):
            q = rand_quat(rng)
            npt.assert_allclose(np.linalg.norm(q), 1.0, atol=1e-6)


# ---------------------------------------------------------------------------
# SLERP interpolation
# ---------------------------------------------------------------------------


class TestQuatInterp:
    def test_integer_indices_exact(self):
        from tsjax.quaternion.transforms import quat_interp

        quats = np.array(
            [[1.0, 0, 0, 0], [np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)]],
            dtype=np.float32,
        )
        ind = np.array([0.0, 1.0])
        result = quat_interp(quats, ind)
        npt.assert_allclose(result[0], quats[0], atol=1e-5)
        npt.assert_allclose(result[1], quats[1], atol=1e-5)

    def test_midpoint(self):
        from tsjax.quaternion.transforms import quat_interp

        # Identity to 90-deg-z — midpoint should be 45-deg-z
        quats = np.array(
            [[1.0, 0, 0, 0], [np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)]],
            dtype=np.float32,
        )
        ind = np.array([0.5])
        result = quat_interp(quats, ind)
        expected = np.array([np.cos(np.pi / 8), 0, 0, np.sin(np.pi / 8)])
        npt.assert_allclose(np.abs(result[0]), np.abs(expected), atol=1e-5)

    def test_nan_out_of_range(self):
        from tsjax.quaternion.transforms import quat_interp

        quats = np.array([[1.0, 0, 0, 0], [0.0, 1, 0, 0]], dtype=np.float32)
        ind = np.array([-0.5, 1.5])
        result = quat_interp(quats, ind, extend=False)
        assert np.all(np.isnan(result[0]))
        assert np.all(np.isnan(result[1]))

    def test_extend_clamps(self):
        from tsjax.quaternion.transforms import quat_interp

        quats = np.array([[1.0, 0, 0, 0], [0.0, 1, 0, 0]], dtype=np.float32)
        ind = np.array([-0.5, 1.5])
        result = quat_interp(quats, ind, extend=True)
        assert not np.any(np.isnan(result))

    def test_unit_norm_output(self):
        from tsjax.quaternion.transforms import quat_interp

        rng = np.random.default_rng(42)
        quats = rng.standard_normal((10, 4)).astype(np.float32)
        quats /= np.linalg.norm(quats, axis=-1, keepdims=True)
        # Use interior indices only (avoid boundary NaN issues)
        ind = np.linspace(0, 9 - 1e-9, 25)
        result = quat_interp(quats, ind)
        norms = np.linalg.norm(result, axis=-1)
        npt.assert_allclose(norms, 1.0, atol=1e-5)


# ---------------------------------------------------------------------------
# Quaternion augmentation
# ---------------------------------------------------------------------------


class TestQuaternionAugmentation:
    def test_preserves_shape(self):
        from tsjax.quaternion.transforms import quaternion_augmentation

        aug = quaternion_augmentation([(0, 2), (3, 5)])
        rng = np.random.default_rng(42)
        u = rng.standard_normal((100, 6)).astype(np.float32)
        y = rng.standard_normal((100, 4)).astype(np.float32)
        y /= np.linalg.norm(y, axis=-1, keepdims=True)
        item = {"u": u, "y": y}
        result = aug(item, rng)
        assert result["u"].shape == u.shape
        assert result["y"].shape == y.shape

    def test_changes_data(self):
        from tsjax.quaternion.transforms import quaternion_augmentation

        aug = quaternion_augmentation([(0, 2)])
        rng = np.random.default_rng(42)
        u = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        y = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        item = {"u": u, "y": y}
        result = aug(item, rng)
        assert not np.allclose(result["u"], u) or not np.allclose(result["y"], y)

    def test_p_zero_unchanged(self):
        from tsjax.quaternion.transforms import quaternion_augmentation

        aug = quaternion_augmentation([(0, 2)], p=0.0)
        rng = np.random.default_rng(42)
        u = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        y = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        item = {"u": u, "y": y}
        result = aug(item, rng)
        npt.assert_array_equal(result["u"], u)
        npt.assert_array_equal(result["y"], y)

    def test_deterministic_seed(self):
        from tsjax.quaternion.transforms import quaternion_augmentation

        aug = quaternion_augmentation([(0, 2)])
        u = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
        y = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float32)
        item = {"u": u.copy(), "y": y.copy()}

        r1 = aug(item, np.random.default_rng(123))
        r2 = aug({"u": u.copy(), "y": y.copy()}, np.random.default_rng(123))
        npt.assert_array_equal(r1["u"], r2["u"])
        npt.assert_array_equal(r1["y"], r2["y"])

    def test_invalid_group_size(self):
        from tsjax.quaternion.transforms import quaternion_augmentation

        with pytest.raises(ValueError, match="expected 3 or 4"):
            quaternion_augmentation([(0, 1)])  # 2 channels — invalid

    def test_4ch_quaternion_group(self):
        from tsjax.quaternion.transforms import quaternion_augmentation

        aug = quaternion_augmentation([(0, 3)])  # 4-ch group = quaternion
        rng = np.random.default_rng(42)
        u = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        y = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        item = {"u": u, "y": y}
        result = aug(item, rng)
        # Both should still be unit quaternions
        npt.assert_allclose(np.linalg.norm(result["u"], axis=-1), 1.0, atol=1e-5)
        npt.assert_allclose(np.linalg.norm(result["y"], axis=-1), 1.0, atol=1e-5)
