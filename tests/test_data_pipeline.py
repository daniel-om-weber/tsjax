"""Integration tests for the data pipeline: stats, HDF5 reading, windowing, batching."""

import h5py
import numpy as np


def test_norm_stats_shape_and_finite(dataset_path):
    """compute_norm_stats returns finite float32 arrays with one element per signal."""
    from tsjax import compute_norm_stats

    train_files = sorted(str(p) for p in (dataset_path / "train").rglob("*.hdf5"))
    mean, std = compute_norm_stats(train_files, ["u"])
    assert mean.shape == (1,)
    assert std.shape == (1,)
    assert mean.dtype == np.float32
    assert np.all(np.isfinite(mean))
    assert np.all(std > 0)


def test_norm_stats_match_manual_computation(dataset_path):
    """Stats should match a direct h5py mean/std calculation."""
    from tsjax import compute_norm_stats

    train_files = sorted(str(p) for p in (dataset_path / "train").rglob("*.hdf5"))
    mean, std = compute_norm_stats(train_files, ["u"])

    # Manual computation
    all_data = []
    for f in train_files:
        with h5py.File(f, "r") as hf:
            all_data.append(hf["u"][:])
    data = np.concatenate(all_data)
    expected_mean = np.mean(data).astype(np.float32)
    expected_std = np.std(data).astype(np.float32)

    np.testing.assert_allclose(mean[0], expected_mean, atol=1e-5)
    np.testing.assert_allclose(std[0], expected_std, atol=1e-5)


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


def test_pipeline_has_norm_stats(pipeline):
    """Pipeline should expose finite norm stats with correct length."""
    assert len(pipeline.u_mean) == 1
    assert len(pipeline.u_std) == 1
    assert len(pipeline.y_mean) == 1
    assert len(pipeline.y_std) == 1
    assert np.all(np.isfinite(pipeline.u_mean))
    assert np.all(pipeline.u_std > 0)


def test_test_split_full_sequences(pipeline):
    """Test split should yield full-length sequences (longer than win_sz)."""
    batch = next(iter(pipeline.test))
    assert batch["u"].shape[0] == 1  # batch size 1
    assert batch["u"].shape[1] > 20  # full sequence, not windowed
    assert batch["u"].shape[2] == 1


def test_window_count_matches_formula(dataset_path):
    """Window count should match: (seq_len - win_sz) // stp_sz + 1."""
    from tsjax import HDF5Store
    from tsjax.data import WindowedSource

    train_files = sorted(str(p) for p in (dataset_path / "train").rglob("*.hdf5"))
    index = HDF5Store(train_files, ["u", "y"])
    source = WindowedSource(index, win_sz=20, stp_sz=10, input_signals=["u"], output_signals=["y"])

    # Verify against formula
    expected_total = 0
    for path in train_files:
        seq_len = index.get_seq_len(path, "u")
        expected_total += max(0, (seq_len - 20) // 10 + 1)

    assert len(source) == expected_total
