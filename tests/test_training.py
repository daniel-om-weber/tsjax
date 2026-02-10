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
