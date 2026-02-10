"""Integration tests for end-to-end training."""

import pytest


@pytest.fixture(scope="module")
def trained_learner(pipeline):
    """Train a small GRU for 3 epochs — shared across tests in this module."""
    from tsjax import RNNLearner, rmse

    lrn = RNNLearner(pipeline, rnn_type="gru", hidden_size=8, seed=42, metrics=[rmse])
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
    from tsjax import RNNLearner

    lrn = RNNLearner(pipeline, rnn_type="gru", hidden_size=8, seed=42)
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

    lrn = RNNLearner(pipeline, rnn_type="lstm", hidden_size=8, seed=42)
    lrn.fit(n_epoch=1, lr=1e-3, progress=False)
    assert len(lrn.train_losses) == 1


def test_n_skip(pipeline):
    """n_skip should not crash and should still produce loss."""
    from tsjax import RNNLearner

    lrn = RNNLearner(pipeline, rnn_type="gru", hidden_size=8, n_skip=5, seed=42)
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


def test_classifier_learner_trains(classification_dataset):
    """ClassifierLearner should train for 1 epoch without error."""
    from tsjax import ClassifierLearner, ScalarAttr, create_grain_dls

    pipeline = create_grain_dls(
        inputs={"u": ["sensor"]},
        targets={"y": ScalarAttr(["fault_class"])},
        dataset=classification_dataset,
        win_sz=20,
        stp_sz=20,
        bs=2,
    )
    lrn = ClassifierLearner(pipeline, n_classes=3, hidden_size=8, seed=42)
    lrn.fit(n_epoch=1, lr=1e-3, progress=False)
    assert len(lrn.train_losses) == 1
    assert lrn.train_losses[0] > 0


def test_classifier_learner_loss_is_cross_entropy(classification_dataset):
    """ClassifierLearner should use cross_entropy_loss."""
    from tsjax import ClassifierLearner, ScalarAttr, create_grain_dls
    from tsjax.losses.classification import cross_entropy_loss

    pipeline = create_grain_dls(
        inputs={"u": ["sensor"]},
        targets={"y": ScalarAttr(["fault_class"])},
        dataset=classification_dataset,
        win_sz=20,
        stp_sz=20,
        bs=2,
    )
    lrn = ClassifierLearner(pipeline, n_classes=3, hidden_size=8)
    assert lrn.loss_func is cross_entropy_loss


def test_regression_learner_trains(regression_dataset):
    """RegressionLearner should train for 1 epoch without error."""
    from tsjax import RegressionLearner, ScalarAttr, create_grain_dls

    pipeline = create_grain_dls(
        inputs={"u": ScalarAttr(["mass", "stiffness"])},
        targets={"y": ScalarAttr(["freq"])},
        dataset=regression_dataset,
        bs=2,
    )
    lrn = RegressionLearner(pipeline, hidden_sizes=[8, 4], seed=42)
    lrn.fit(n_epoch=1, lr=1e-3, progress=False)
    assert len(lrn.train_losses) == 1
    assert lrn.train_losses[0] > 0


def test_regression_learner_model_is_mlp(regression_dataset):
    """RegressionLearner should use MLP model."""
    from tsjax import MLP, RegressionLearner, ScalarAttr, create_grain_dls

    pipeline = create_grain_dls(
        inputs={"u": ScalarAttr(["mass", "stiffness"])},
        targets={"y": ScalarAttr(["freq"])},
        dataset=regression_dataset,
        bs=2,
    )
    lrn = RegressionLearner(pipeline, hidden_sizes=[8])
    from tsjax import NormalizedModel

    assert isinstance(lrn.model, NormalizedModel)
    assert isinstance(lrn.model.model, MLP)
