"""Tests for visualization functions."""

import matplotlib.pyplot as plt
import numpy as np

from tsjax.viz import (
    plot_batch,
    plot_classification_results,
    plot_regression_scatter,
    plot_results,
    plot_scalar_batch,
)


class TestPlotBatch:
    def test_basic_shape(self):
        batch = {"u": np.random.randn(8, 50, 1), "y": np.random.randn(8, 50, 1)}
        fig, axes = plot_batch(batch, n=4)
        assert axes.shape == (2, 4)
        plt.close(fig)

    def test_n_clamped_to_batch_size(self):
        batch = {"u": np.random.randn(2, 50, 1), "y": np.random.randn(2, 50, 1)}
        fig, axes = plot_batch(batch, n=10)
        assert axes.shape == (2, 2)
        plt.close(fig)

    def test_multi_channel(self):
        batch = {"u": np.random.randn(4, 50, 3), "y": np.random.randn(4, 50, 2)}
        fig, axes = plot_batch(batch, n=2, u_labels=["a", "b", "c"], y_labels=["x", "z"])
        assert axes.shape == (2, 2)
        plt.close(fig)


class TestPlotResults:
    def test_with_u(self):
        target = np.random.randn(8, 50, 1)
        pred = np.random.randn(8, 50, 1)
        u = np.random.randn(8, 50, 1)
        fig, axes = plot_results(target, pred, n=4, u=u)
        assert axes.shape == (2, 4)  # 1 u row + 1 y row
        plt.close(fig)

    def test_without_u(self):
        target = np.random.randn(8, 50, 1)
        pred = np.random.randn(8, 50, 1)
        fig, axes = plot_results(target, pred, n=4)
        assert axes.shape == (1, 4)
        plt.close(fig)

    def test_multi_output(self):
        target = np.random.randn(4, 50, 3)
        pred = np.random.randn(4, 50, 3)
        fig, axes = plot_results(target, pred, n=2)
        assert axes.shape == (3, 2)
        plt.close(fig)

    def test_multi_output_with_u(self):
        target = np.random.randn(4, 50, 2)
        pred = np.random.randn(4, 50, 2)
        u = np.random.randn(4, 50, 1)
        fig, axes = plot_results(target, pred, n=3, u=u)
        assert axes.shape == (3, 3)  # 1 u row + 2 y rows, 3 samples
        plt.close(fig)


class TestLearnerIntegration:
    def test_show_batch(self, pipeline):
        from tsjax import RNNLearner

        lrn = RNNLearner(pipeline, hidden_size=8, seed=42)
        fig, axes = lrn.show_batch(n=2, split="valid")
        assert axes.shape == (2, 2)  # 2 signal rows, 2 samples
        plt.close(fig)

    def test_show_results(self, pipeline):
        from tsjax import RNNLearner

        lrn = RNNLearner(pipeline, hidden_size=8, seed=42)
        fig, axes = lrn.show_results(n=2, split="valid")
        assert axes.shape == (2, 2)  # u row + y row, 2 samples
        plt.close(fig)

    def test_show_results_test_split(self, pipeline):
        from tsjax import RNNLearner

        lrn = RNNLearner(pipeline, hidden_size=8, seed=42)
        fig, axes = lrn.show_results(n=4, split="test")
        assert axes.shape[1] == 1  # test has bs=1
        plt.close(fig)


# ---------------------------------------------------------------------------
# Phase 7 — new viz functions
# ---------------------------------------------------------------------------


class TestPlotScalarBatch:
    def test_basic_shape(self):
        batch = {"u": np.random.randn(8, 50, 1), "y": np.random.randn(8, 2)}
        fig, axes = plot_scalar_batch(batch, n=4)
        assert axes.shape == (1, 4)
        plt.close(fig)

    def test_n_clamped(self):
        batch = {"u": np.random.randn(2, 50, 1), "y": np.random.randn(2, 1)}
        fig, axes = plot_scalar_batch(batch, n=10)
        assert axes.shape == (1, 2)
        plt.close(fig)

    def test_with_labels(self):
        batch = {"u": np.random.randn(4, 30, 2), "y": np.random.randn(4, 1)}
        fig, axes = plot_scalar_batch(batch, n=2, u_labels=["a", "b"], y_labels=["class"])
        assert axes.shape == (1, 2)
        plt.close(fig)


class TestPlotClassificationResults:
    def test_basic_shape(self):
        target = np.array([0, 1, 2, 0])
        pred = np.random.randn(4, 3)
        fig, axes = plot_classification_results(target, pred, n=3)
        assert axes.shape == (1, 3)
        plt.close(fig)

    def test_2d_target(self):
        target = np.array([[0], [1], [2]])
        pred = np.random.randn(3, 3)
        fig, axes = plot_classification_results(target, pred, n=2)
        assert axes.shape == (1, 2)
        plt.close(fig)

    def test_with_class_names(self):
        target = np.array([0, 1])
        pred = np.random.randn(2, 2)
        fig, axes = plot_classification_results(target, pred, n=2, class_names=["cat", "dog"])
        assert axes.shape == (1, 2)
        plt.close(fig)


class TestPlotRegressionScatter:
    def test_basic_shape(self):
        target = np.random.randn(20, 1)
        pred = np.random.randn(20, 1)
        fig, axes = plot_regression_scatter(target, pred)
        assert axes.shape == (1, 1)
        plt.close(fig)

    def test_multi_output(self):
        target = np.random.randn(20, 3)
        pred = np.random.randn(20, 3)
        fig, axes = plot_regression_scatter(target, pred)
        assert axes.shape == (1, 3)
        plt.close(fig)

    def test_1d_input(self):
        target = np.random.randn(10)
        pred = np.random.randn(10)
        fig, axes = plot_regression_scatter(target, pred)
        assert axes.shape == (1, 1)
        plt.close(fig)

    def test_n_limit(self):
        target = np.random.randn(100, 2)
        pred = np.random.randn(100, 2)
        fig, axes = plot_regression_scatter(target, pred, n=20)
        assert axes.shape == (1, 2)
        plt.close(fig)

    def test_with_labels(self):
        target = np.random.randn(10, 2)
        pred = np.random.randn(10, 2)
        fig, axes = plot_regression_scatter(target, pred, y_labels=["freq", "amp"])
        assert axes.shape == (1, 2)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Phase 7 — Learner callback integration
# ---------------------------------------------------------------------------


class TestLearnerCallbacks:
    def test_custom_plot_batch_fn(self, pipeline):
        from flax import nnx

        from tsjax import RNN, Learner

        model = RNN(
            input_size=1,
            output_size=1,
            hidden_size=8,
            rngs=nnx.Rngs(0),
        )
        called = {}

        def custom_batch_fn(batch, *, n, figsize, source, pipeline):
            called["yes"] = True
            fig, axes = plt.subplots(1, 1)
            return fig, np.array([[axes]])

        lrn = Learner(model, pipeline, plot_batch_fn=custom_batch_fn)
        fig, _ = lrn.show_batch(n=1)
        plt.close(fig)
        assert "yes" in called

    def test_custom_plot_results_fn(self, pipeline):
        from flax import nnx

        from tsjax import RNN, Learner

        model = RNN(
            input_size=1,
            output_size=1,
            hidden_size=8,
            rngs=nnx.Rngs(0),
        )
        called = {}

        def custom_results_fn(*, target, pred, n, figsize, batch, source, pipeline):
            called["yes"] = True
            fig, axes = plt.subplots(1, 1)
            return fig, np.array([[axes]])

        lrn = Learner(model, pipeline, plot_results_fn=custom_results_fn)
        fig, _ = lrn.show_results(n=1)
        plt.close(fig)
        assert "yes" in called
