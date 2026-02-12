"""Integration tests for end-to-end training."""

import pytest


@pytest.fixture(scope="module")
def trained_learner(pipeline):
    """Train a small GRU for 3 epochs — shared across tests in this module."""
    from tsjax import GRULearner, rmse

    lrn = GRULearner(pipeline, hidden_size=8, seed=42, metrics=[rmse])
    lrn.fit(n_epoch=3, lr=1e-3, progress=False)
    return lrn


def test_training_reduces_loss(trained_learner):
    """After 3 epochs, train loss should decrease."""
    losses = trained_learner.train_losses
    assert len(losses) == 3
    assert losses[-1] < losses[0]


def test_valid_loss_tracked(trained_learner):
    """Validation loss should be tracked each epoch."""
    assert len(trained_learner.valid_losses) == 3
    assert all(v > 0 for v in trained_learner.valid_losses)


def test_metrics_tracked(trained_learner):
    """Passing metrics=[rmse] should populate valid_metrics."""
    assert "rmse" in trained_learner.valid_metrics
    assert len(trained_learner.valid_metrics["rmse"]) == 3
    assert all(v > 0 for v in trained_learner.valid_metrics["rmse"])


def test_fit_flat_cos(pipeline):
    """fit_flat_cos schedule should also reduce loss."""
    from tsjax import GRULearner

    lrn = GRULearner(pipeline, hidden_size=8, seed=42)
    lrn.fit_flat_cos(n_epoch=3, lr=1e-3, progress=False)
    assert lrn.train_losses[-1] < lrn.train_losses[0]


def test_gru_learner_factory(pipeline):
    """GRULearner factory should produce a working learner."""
    from tsjax import GRULearner

    lrn = GRULearner(pipeline, hidden_size=8, seed=42)
    lrn.fit(n_epoch=1, lr=1e-3, progress=False)
    assert len(lrn.train_losses) == 1
    assert lrn.train_losses[0] > 0


def test_lstm_learner(pipeline):
    """LSTM variant should train without error."""
    from tsjax import RNNLearner

    lrn = RNNLearner(pipeline, hidden_size=8, seed=42)
    lrn.fit(n_epoch=1, lr=1e-3, progress=False)
    assert len(lrn.train_losses) == 1


def test_n_skip(pipeline):
    """n_skip should not crash and should still produce loss."""
    from tsjax import GRULearner

    lrn = GRULearner(pipeline, hidden_size=8, n_skip=5, seed=42)
    lrn.fit(n_epoch=1, lr=1e-3, progress=False)
    assert lrn.train_losses[0] > 0


# ---------------------------------------------------------------------------
# Phase 8 — ClassifierLearner and RegressionLearner factories
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def classification_dataset(tmp_path_factory):
    """Dataset with integer class label HDF5 attributes for classification."""
    import h5py
    import numpy as np

    tmp = tmp_path_factory.mktemp("cls_data")
    rng = np.random.default_rng(42)
    for split in ("train", "valid", "test"):
        d = tmp / split
        d.mkdir()
        for i in range(6):
            with h5py.File(str(d / f"f{i}.hdf5"), "w") as f:
                f.create_dataset("sensor", data=rng.standard_normal(100).astype(np.float32))
                f.attrs["fault_class"] = float(rng.integers(0, 3))
    return tmp


@pytest.fixture(scope="module")
def regression_dataset(tmp_path_factory):
    """Dataset with scalar HDF5 attributes for tabular regression."""
    import h5py
    import numpy as np

    tmp = tmp_path_factory.mktemp("reg_data")
    rng = np.random.default_rng(42)
    for split in ("train", "valid", "test"):
        d = tmp / split
        d.mkdir()
        for i in range(6):
            with h5py.File(str(d / f"f{i}.hdf5"), "w") as f:
                f.attrs["mass"] = float(rng.standard_normal())
                f.attrs["stiffness"] = float(rng.standard_normal())
                f.attrs["freq"] = float(rng.standard_normal())
    return tmp


def _make_split_files(dataset_path):
    """Get sorted HDF5 files per split."""
    return {
        split: sorted(str(p) for p in (dataset_path / split).rglob("*.hdf5"))
        for split in ("train", "valid", "test")
    }


def _make_classifier_pipeline(classification_dataset):
    """Build a classifier pipeline with scalar target via manual construction."""
    from tsjax import GrainPipeline, HDF5Store, WindowedSource
    from tsjax.data.sources import FileSource, scalar_attrs

    sf = _make_split_files(classification_dataset)

    def make_source(split, windowed=True):
        files = sf[split]
        store = HDF5Store(files, ["sensor"])
        specs = {"u": ["sensor"], "y": scalar_attrs(files, ["fault_class"])}
        if windowed:
            return WindowedSource(store, specs, win_sz=20, stp_sz=20)
        return FileSource(store, specs)

    return GrainPipeline.from_sources(
        make_source("train"),
        make_source("valid"),
        make_source("test", windowed=False),
        input_keys=("u",),
        target_keys=("y",),
        bs=2,
        seed=42,
    )


def _make_regression_pipeline(regression_dataset):
    """Build a regression pipeline with pure-scalar inputs/targets."""
    from tsjax import GrainPipeline, HDF5Store
    from tsjax.data.sources import FileSource, scalar_attrs

    sf = _make_split_files(regression_dataset)

    def make_source(files):
        store = HDF5Store(files, [])
        specs = {
            "u": scalar_attrs(files, ["mass", "stiffness"]),
            "y": scalar_attrs(files, ["freq"]),
        }
        return FileSource(store, specs)

    return GrainPipeline.from_sources(
        make_source(sf["train"]),
        make_source(sf["valid"]),
        make_source(sf["test"]),
        input_keys=("u",),
        target_keys=("y",),
        bs=2,
        seed=42,
    )


def test_classifier_learner_trains(classification_dataset):
    """ClassifierLearner should train for 1 epoch without error."""
    from tsjax import ClassifierLearner

    pipeline = _make_classifier_pipeline(classification_dataset)
    lrn = ClassifierLearner(pipeline, n_classes=3, hidden_size=8, seed=42)
    lrn.fit(n_epoch=1, lr=1e-3, progress=False)
    assert len(lrn.train_losses) == 1
    assert lrn.train_losses[0] > 0


def test_classifier_learner_loss_is_cross_entropy(classification_dataset):
    """ClassifierLearner should use cross_entropy_loss."""
    from tsjax import ClassifierLearner
    from tsjax.losses.classification import cross_entropy_loss

    pipeline = _make_classifier_pipeline(classification_dataset)
    lrn = ClassifierLearner(pipeline, n_classes=3, hidden_size=8)
    assert lrn.loss_func is cross_entropy_loss


def test_regression_learner_trains(regression_dataset):
    """RegressionLearner should train for 1 epoch without error."""
    from tsjax import RegressionLearner

    pipeline = _make_regression_pipeline(regression_dataset)
    lrn = RegressionLearner(pipeline, hidden_sizes=[8, 4], seed=42)
    lrn.fit(n_epoch=1, lr=1e-3, progress=False)
    assert len(lrn.train_losses) == 1
    assert lrn.train_losses[0] > 0


def test_regression_learner_model_is_mlp(regression_dataset):
    """RegressionLearner should use MLP model."""
    from tsjax import MLP, NormalizedModel, RegressionLearner

    pipeline = _make_regression_pipeline(regression_dataset)
    lrn = RegressionLearner(pipeline, hidden_sizes=[8])
    assert isinstance(lrn.model, NormalizedModel)
    assert isinstance(lrn.model.model, MLP)
