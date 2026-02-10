"""Factory functions mirroring TSFast's Learner pattern."""

from __future__ import annotations

from functools import partial

from flax import nnx

from tsjax.data import GrainPipeline
from tsjax.losses import normalized_mae, normalized_mse
from tsjax.losses.classification import cross_entropy_loss
from tsjax.models import MLP, RNN, Denormalize, LastPool, Normalize, NormalizedModel

from .learner import Learner


def create_rnn(
    pipeline: GrainPipeline,
    cell_type: type = nnx.GRUCell,
    hidden_size: int = 100,
    num_layers: int = 1,
    seed: int = 0,
) -> NormalizedModel:
    """Create RNN model with norm stats inferred from pipeline."""
    u_stats = pipeline.stats[pipeline.input_keys[0]]
    y_stats = pipeline.stats[pipeline.target_keys[0]]
    input_size = len(u_stats.mean)
    output_size = len(y_stats.mean)

    rnn = RNN(
        input_size=input_size,
        output_size=output_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        cell_type=cell_type,
        rngs=nnx.Rngs(seed),
    )
    return NormalizedModel(
        rnn,
        norm_in=Normalize(input_size, u_stats.mean, u_stats.std),
        norm_out=Denormalize(output_size, y_stats.mean, y_stats.std),
    )


def RNNLearner(
    pipeline: GrainPipeline,
    cell_type: type = nnx.OptimizedLSTMCell,
    hidden_size: int = 100,
    num_layers: int = 1,
    loss_func=normalized_mae,
    n_skip: int = 0,
    seed: int = 0,
    metrics: list = [],
) -> Learner:
    """Create Learner with RNN model (mirrors TSFast's RNNLearner)."""
    model = create_rnn(
        pipeline, cell_type=cell_type, hidden_size=hidden_size, num_layers=num_layers, seed=seed
    )
    return Learner(model, pipeline, loss_func=loss_func, n_skip=n_skip, metrics=metrics)


create_gru = partial(create_rnn, cell_type=nnx.GRUCell)
GRULearner = partial(RNNLearner, cell_type=nnx.GRUCell)


# ---------------------------------------------------------------------------
# Classification factory
# ---------------------------------------------------------------------------


def _classifier_show_batch(batch, *, n, figsize, source, pipeline):
    from tsjax.viz import plot_scalar_batch

    input_key = pipeline.input_keys[0]
    target_key = pipeline.target_keys[0]
    return plot_scalar_batch(
        batch,
        n=n,
        figsize=figsize,
        u_labels=source.readers[input_key].signals,
        y_labels=source.readers[target_key].signals,
    )


def _classifier_show_results(*, target, pred, n, figsize, batch, source, pipeline):
    from tsjax.viz import plot_classification_results

    return plot_classification_results(target, pred, n=n, figsize=figsize)


def ClassifierLearner(
    pipeline: GrainPipeline,
    n_classes: int,
    cell_type: type = nnx.GRUCell,
    hidden_size: int = 100,
    num_layers: int = 1,
    n_skip: int = 0,
    seed: int = 0,
    metrics: list = [],
) -> Learner:
    """Create Learner with RNN + LastPool model + cross-entropy loss."""
    u_stats = pipeline.stats[pipeline.input_keys[0]]
    input_size = len(u_stats.mean)

    rnn = RNN(
        input_size=input_size,
        output_size=n_classes,
        hidden_size=hidden_size,
        num_layers=num_layers,
        cell_type=cell_type,
        rngs=nnx.Rngs(seed),
    )
    encoder = nnx.Sequential(rnn, LastPool())
    model = NormalizedModel(
        encoder,
        norm_in=Normalize(input_size, u_stats.mean, u_stats.std),
        norm_out=Denormalize(n_classes),  # identity — logits pass through
    )
    return Learner(
        model,
        pipeline,
        loss_func=cross_entropy_loss,
        n_skip=n_skip,
        metrics=metrics,
        plot_batch_fn=_classifier_show_batch,
        plot_results_fn=_classifier_show_results,
    )


# ---------------------------------------------------------------------------
# Regression (MLP) factory
# ---------------------------------------------------------------------------


def _regression_show_results(*, target, pred, n, figsize, batch, source, pipeline):
    from tsjax.viz import plot_regression_scatter

    target_key = pipeline.target_keys[0]
    return plot_regression_scatter(
        target,
        pred,
        n=n,
        figsize=figsize,
        y_labels=source.readers[target_key].signals,
    )


def RegressionLearner(
    pipeline: GrainPipeline,
    hidden_sizes: list[int] | None = None,
    loss_func=normalized_mse,
    seed: int = 0,
    metrics: list = [],
) -> Learner:
    """Create Learner with MLP model for scalar regression."""
    u_stats = pipeline.stats[pipeline.input_keys[0]]
    y_stats = pipeline.stats[pipeline.target_keys[0]]
    input_size = len(u_stats.mean)
    output_size = len(y_stats.mean)

    mlp = MLP(
        input_size=input_size,
        output_size=output_size,
        hidden_sizes=hidden_sizes,
        rngs=nnx.Rngs(seed),
    )
    model = NormalizedModel(
        mlp,
        norm_in=Normalize(input_size, u_stats.mean, u_stats.std),
        norm_out=Denormalize(output_size, y_stats.mean, y_stats.std),
    )
    return Learner(
        model,
        pipeline,
        loss_func=loss_func,
        n_skip=0,
        metrics=metrics,
        plot_results_fn=_regression_show_results,
    )
