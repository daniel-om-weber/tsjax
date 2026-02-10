"""Learner class with fit methods mirroring fastai's API."""

from __future__ import annotations

import time

import jax.numpy as jnp
import optax
from flax import nnx
from tqdm import tqdm

from .models import RNN
from .pipeline import GrainPipeline
from .train import normalized_mae


class Learner:
    """Wraps an RNN model and pipeline with fastai-style fit methods."""

    def __init__(
        self,
        model: RNN,
        pipeline: GrainPipeline,
        loss_func=normalized_mae,
        n_skip: int = 0,
        metrics: list = [],
    ):
        self.model = model
        self.pipeline = pipeline
        self.loss_func = loss_func
        self.n_skip = n_skip
        self.metrics = metrics
        self.train_losses: list[float] = []
        self.valid_losses: list[float] = []
        self.valid_metrics: dict[str, list[float]] = {m.__name__: [] for m in metrics}

    def fit(self, n_epoch: int, lr: float = 3e-3, progress: bool = True):
        """Train with constant LR."""
        tx = optax.adam(lr)
        self._fit(n_epoch, tx, progress=progress)

    def fit_flat_cos(
        self, n_epoch: int, lr: float = 3e-3, pct_start: float = 0.75, progress: bool = True
    ):
        """Train with flat LR then cosine decay."""
        batches_per_epoch = len(self.pipeline.train)
        total_steps = n_epoch * batches_per_epoch
        flat_steps = int(total_steps * pct_start)
        decay_steps = total_steps - flat_steps

        schedule = optax.join_schedules(
            schedules=[
                optax.constant_schedule(lr),
                optax.cosine_decay_schedule(init_value=lr, decay_steps=max(decay_steps, 1)),
            ],
            boundaries=[flat_steps],
        )
        tx = optax.adam(learning_rate=schedule)
        self._fit(n_epoch, tx, progress=progress)

    def _fit(self, n_epoch: int, tx, progress: bool = True):
        """Internal training loop."""
        y_mean = jnp.asarray(self.pipeline.y_mean)
        y_std = jnp.asarray(self.pipeline.y_std)
        n_skip = self.n_skip
        loss_func = self.loss_func
        metric_fns = list(self.metrics)
        model = self.model

        optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)

        def loss_fn(model, u, y):
            pred = model(u)
            if n_skip > 0:
                pred = pred[:, n_skip:]
                y = y[:, n_skip:]
            return loss_func(pred, y, y_mean, y_std)

        @nnx.jit
        def train_step(model, optimizer, u, y):
            loss, grads = nnx.value_and_grad(loss_fn)(model, u, y)
            optimizer.update(model, grads)
            return loss

        @nnx.jit
        def eval_step(model, u, y):
            pred = model(u)
            if n_skip > 0:
                pred = pred[:, n_skip:]
                y = y[:, n_skip:]
            loss = loss_func(pred, y, y_mean, y_std)
            mvals = [m(pred, y, y_mean, y_std) for m in metric_fns]
            return loss, mvals

        n_total_batches = len(self.pipeline.train)

        for epoch in range(n_epoch):
            epoch_start = time.time()
            # Training
            epoch_loss = 0.0
            n_batches = 0
            batch_iter = self.pipeline.train
            if progress:
                batch_iter = tqdm(
                    batch_iter,
                    total=n_total_batches,
                    desc=f"Epoch {epoch + 1}/{n_epoch}",
                    leave=False,
                    mininterval=1.0,
                )
            for batch in batch_iter:
                u = jnp.asarray(batch["u"])
                y = jnp.asarray(batch["y"])
                loss = train_step(model, optimizer, u, y)
                epoch_loss += float(loss)
                n_batches += 1
                if progress:
                    batch_iter.set_postfix(loss=f"{epoch_loss / n_batches:.4f}", refresh=False)

            avg_train = epoch_loss / max(n_batches, 1)
            self.train_losses.append(avg_train)

            # Validation
            val_loss = 0.0
            metric_sums = [0.0] * len(metric_fns)
            n_val = 0
            for batch in self.pipeline.valid:
                u = jnp.asarray(batch["u"])
                y = jnp.asarray(batch["y"])
                loss, mvals = eval_step(model, u, y)
                val_loss += float(loss)
                for i, v in enumerate(mvals):
                    metric_sums[i] += float(v)
                n_val += 1

            avg_val = val_loss / max(n_val, 1)
            self.valid_losses.append(avg_val)
            metric_avgs = [s / max(n_val, 1) for s in metric_sums]
            for mf, avg in zip(metric_fns, metric_avgs):
                self.valid_metrics[mf.__name__].append(avg)

            t = int(time.time() - epoch_start)
            h, m, s = t // 3600, (t // 60) % 60, t % 60
            time_str = f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"

            total_epoch = len(self.train_losses)
            metrics_str = "".join(
                f"  {mf.__name__}={avg:.6f}" for mf, avg in zip(metric_fns, metric_avgs)
            )
            print(
                f"Epoch {total_epoch:3d}  train_loss={avg_train:.6f}"
                f"  valid_loss={avg_val:.6f}{metrics_str}  {time_str}"
            )
