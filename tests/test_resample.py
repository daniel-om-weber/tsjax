"""Tests for resampling functions and ResampledStore wrapper."""

from __future__ import annotations

import pickle
from pathlib import Path

import grain
import h5py
import numpy as np
import pytest

from tsjax.data.hdf5_store import HDF5Store, read_hdf5_attr
from tsjax.data.resample import ResampledStore, resample_fft, resample_interp
from tsjax.data.sources import DataSource, SequenceReader
from tsjax.data.stats import compute_stats

DATASET = Path(__file__).parent.parent / "test_data" / "WienerHammerstein"
TRAIN_DIR = DATASET / "train"
ALL_SIGNALS = ["u", "y"]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def train_store():
    files = sorted(str(p) for p in TRAIN_DIR.rglob("*") if p.suffix in {".hdf5", ".h5"})
    return HDF5Store(files, ALL_SIGNALS, preload=True)


@pytest.fixture()
def sine_signal():
    """A 1000-sample sine wave at 10 Hz sampled at 1000 Hz."""
    t = np.linspace(0, 1, 1000, endpoint=False)
    return np.sin(2 * np.pi * 10 * t).astype(np.float32)


# ---------------------------------------------------------------------------
# Pure function tests
# ---------------------------------------------------------------------------


class TestResampleInterp:
    def test_identity(self, sine_signal):
        result = resample_interp(sine_signal, 1.0)
        np.testing.assert_array_equal(result, sine_signal)

    def test_downsample_length(self, sine_signal):
        result = resample_interp(sine_signal, 0.5)
        assert len(result) == 500

    def test_upsample_length(self, sine_signal):
        result = resample_interp(sine_signal, 2.0)
        assert len(result) == 2000

    def test_preserves_dtype(self, sine_signal):
        result = resample_interp(sine_signal, 0.5)
        assert result.dtype == sine_signal.dtype

    def test_zero_length_raises(self, sine_signal):
        with pytest.raises(ValueError, match="zero-length"):
            resample_interp(sine_signal, 0.0)

    def test_anti_aliasing_applied(self):
        """Downsampling with anti-aliasing should attenuate high frequencies."""
        # Signal with both low and high frequency content
        t = np.linspace(0, 1, 1000, endpoint=False)
        low = np.sin(2 * np.pi * 5 * t)
        high = np.sin(2 * np.pi * 400 * t)
        sig = (low + high).astype(np.float32)

        result = resample_interp(sig, 0.1, lowpass_cut=1.0)
        # After 10x downsampling with anti-aliasing, the high-freq component
        # should be heavily attenuated.  The result should mostly contain the
        # low-freq component.
        assert len(result) == 100
        # Reconstruct what a pure 5 Hz sine looks like at the lower rate
        t_low = np.linspace(0, 1, 100, endpoint=False)
        expected_low = np.sin(2 * np.pi * 5 * t_low)
        # Correlation with low-freq component should be high
        corr = np.corrcoef(result, expected_low)[0, 1]
        assert corr > 0.9


class TestResampleFFT:
    def test_identity(self, sine_signal):
        result = resample_fft(sine_signal, 1.0)
        np.testing.assert_array_equal(result, sine_signal)

    def test_downsample_length(self, sine_signal):
        result = resample_fft(sine_signal, 0.5)
        assert len(result) == 500

    def test_upsample_length(self, sine_signal):
        result = resample_fft(sine_signal, 2.0)
        assert len(result) == 2000

    def test_preserves_dtype(self, sine_signal):
        result = resample_fft(sine_signal, 0.5)
        assert result.dtype == sine_signal.dtype

    def test_zero_length_raises(self, sine_signal):
        with pytest.raises(ValueError, match="zero-length"):
            resample_fft(sine_signal, 0.0)


# ---------------------------------------------------------------------------
# ResampledStore tests
# ---------------------------------------------------------------------------


class TestResampledStore:
    def test_paths_delegates(self, train_store):
        rs = ResampledStore(train_store, factor=0.5)
        assert rs.paths == train_store.paths

    def test_get_seq_len_uniform(self, train_store):
        rs = ResampledStore(train_store, factor=0.5)
        path = rs.paths[0]
        orig = train_store.get_seq_len(path)
        assert rs.get_seq_len(path) == round(orig * 0.5)

    def test_get_seq_len_callable(self, train_store):
        rs = ResampledStore(train_store, factor=lambda _p: 0.25)
        path = rs.paths[0]
        orig = train_store.get_seq_len(path)
        assert rs.get_seq_len(path) == round(orig * 0.25)

    def test_read_signals_shape(self, train_store):
        rs = ResampledStore(train_store, factor=0.5)
        path = rs.paths[0]
        result = rs.read_signals(path, ["u", "y"], 0, 100)
        assert result.shape == (100, 2)

    def test_read_signals_matches_full_resample(self, train_store):
        """Slicing from ResampledStore should match resample-then-slice."""
        rs = ResampledStore(train_store, factor=0.5)
        path = rs.paths[0]

        # Read a window via ResampledStore
        window = rs.read_signals(path, ["u"], 10, 30)

        # Manually resample full signal and slice
        orig_len = train_store.get_seq_len(path, "u")
        full_raw = train_store.read_signals(path, ["u"], 0, orig_len)[:, 0]
        full_resampled = resample_interp(full_raw, 0.5)
        expected = full_resampled[10:30]

        np.testing.assert_array_equal(window[:, 0], expected)

    def test_picklability(self, train_store):
        rs = ResampledStore(train_store, factor=0.5)
        path = rs.paths[0]
        # Populate cache
        rs.read_signals(path, ["u"], 0, 10)

        # Pickle round-trip
        data = pickle.dumps(rs)
        rs2 = pickle.loads(data)

        # Cache should be empty after unpickle but reads should still work
        assert rs2._cache == {}
        result = rs2.read_signals(path, ["u"], 0, 10)
        assert result.shape == (10, 1)

    def test_with_windowed_source(self, train_store):
        """DataSource with windowing should work with ResampledStore."""
        factor = 0.5
        rs = ResampledStore(train_store, factor=factor)
        win_sz = 20
        stp_sz = 10

        readers = {"u": SequenceReader(rs, ["u"]), "y": SequenceReader(rs, ["y"])}
        source = DataSource(rs, readers, win_sz=win_sz, stp_sz=stp_sz)

        # Verify window count uses resampled lengths
        path = rs.paths[0]
        resampled_len = rs.get_seq_len(path)
        expected_n_win = max(0, (resampled_len - win_sz) // stp_sz + 1)
        assert len(source) == expected_n_win

        # First item should have correct shape
        item = source[0]
        assert item["u"].shape == (win_sz, 1)
        assert item["y"].shape == (win_sz, 1)

    def test_with_full_sequence_source(self, train_store):
        rs = ResampledStore(train_store, factor=0.5)
        readers = {"u": SequenceReader(rs, ["u"]), "y": SequenceReader(rs, ["y"])}
        source = DataSource(rs, readers)
        item = source[0]
        expected_len = rs.get_seq_len(rs.paths[0])
        assert item["u"].shape == (expected_len, 1)
        assert item["y"].shape == (expected_len, 1)


# ---------------------------------------------------------------------------
# read_hdf5_attr tests
# ---------------------------------------------------------------------------


class TestReadHDF5Attr:
    def test_read_attr(self, tmp_path):
        path = str(tmp_path / "test.hdf5")
        with h5py.File(path, "w") as f:
            f.attrs["sampling_rate"] = 100.0
        result = read_hdf5_attr(path, "sampling_rate")
        assert float(result) == pytest.approx(100.0)

    def test_missing_attr_raises(self, tmp_path):
        path = str(tmp_path / "test.hdf5")
        with h5py.File(path, "w") as f:
            f.attrs["other"] = 1.0
        with pytest.raises(KeyError):
            read_hdf5_attr(path, "sampling_rate")


# ---------------------------------------------------------------------------
# compute_norm_stats_from_index tests
# ---------------------------------------------------------------------------


class TestComputeNormStatsFromIndex:
    @staticmethod
    def _make_loader(store, bs=4):
        readers = {"u": SequenceReader(store, ["u"]), "y": SequenceReader(store, ["y"])}
        source = DataSource(store, readers)
        return (
            grain.MapDataset.source(source)
            .batch(bs, drop_remainder=False)
            .to_iter_dataset()
        )

    def test_matches_original(self, train_store):
        """Store-based stats via compute_stats should match direct h5py computation."""
        dl = self._make_loader(train_store)
        result = compute_stats(dl, ["u"], n_batches=1000)
        ns = result["u"]

        # Manual h5py reference
        all_data = []
        for p in sorted(str(p) for p in TRAIN_DIR.rglob("*") if p.suffix in {".hdf5", ".h5"}):
            import h5py

            with h5py.File(p, "r") as f:
                all_data.append(f["u"][:])
        data = np.concatenate(all_data)
        expected_mean = np.mean(data).astype(np.float32)
        expected_std = np.std(data).astype(np.float32)

        np.testing.assert_allclose(ns.mean[0], expected_mean, atol=1e-5)
        np.testing.assert_allclose(ns.std[0], expected_std, atol=1e-5)

    def test_with_resampled_store(self, train_store):
        """Stats from a ResampledStore should return valid arrays."""
        rs = ResampledStore(train_store, factor=0.5)
        dl = self._make_loader(rs)
        result = compute_stats(dl, ["u"], n_batches=1000)
        ns = result["u"]
        assert ns.mean.shape == (1,)
        assert ns.std.shape == (1,)
        assert np.all(ns.std > 0)


# ---------------------------------------------------------------------------
# Pipeline integration
# ---------------------------------------------------------------------------


class TestPipelineResampling:
    def test_create_grain_dls_with_factor(self):
        from tsjax.data.pipeline import create_simulation_dls

        pl = create_simulation_dls(
            u=["u"],
            y=["y"],
            dataset=DATASET,
            win_sz=20,
            stp_sz=10,
            bs=4,
            preload=True,
            resampling_factor=0.5,
        )
        # Verify we can iterate and shapes are correct
        batch = next(iter(pl.train))
        assert batch["u"].shape[1] == 20  # win_sz preserved
        assert batch["y"].shape[1] == 20
