"""Factory functions mirroring TSFast's Learner pattern."""
from __future__ import annotations

from functools import partial

import jax.numpy as jnp
from flax import nnx

from .models import RNN
from .pipeline import GrainPipeline
from .learner import Learner
from .train import normalized_mae


def create_rnn(
    pipeline: GrainPipeline,
    rnn_type: str = 'gru',
    hidden_size: int = 100,
    num_layers: int = 1,
    seed: int = 0,
) -> RNN:
    """Create RNN model with norm stats inferred from pipeline."""
    input_size = len(pipeline.u_mean)
    output_size = len(pipeline.y_mean)

    return RNN(
        input_size=input_size,
        output_size=output_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        rnn_type=rnn_type,
        u_mean=jnp.asarray(pipeline.u_mean),
        u_std=jnp.asarray(pipeline.u_std),
        y_mean=jnp.asarray(pipeline.y_mean),
        y_std=jnp.asarray(pipeline.y_std),
        rngs=nnx.Rngs(seed),
    )


def RNNLearner(
    pipeline: GrainPipeline,
    rnn_type: str = 'lstm',
    hidden_size: int = 100,
    num_layers: int = 1,
    loss_func=normalized_mae,
    n_skip: int = 0,
    seed: int = 0,
    metrics: list = [],
) -> Learner:
    """Create Learner with RNN model (mirrors TSFast's RNNLearner)."""
    model = create_rnn(pipeline, rnn_type=rnn_type, hidden_size=hidden_size,
                       num_layers=num_layers, seed=seed)
    return Learner(model, pipeline, loss_func=loss_func, n_skip=n_skip, metrics=metrics)


create_gru = partial(create_rnn, rnn_type='gru')  # backward compat
GRULearner = partial(RNNLearner, rnn_type='gru')   # backward compat
