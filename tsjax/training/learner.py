"""Learner class with fit methods mirroring fastai's API."""

from __future__ import annotations

import time

import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx
from tqdm import tqdm

from tsjax.data import GrainPipeline
from tsjax.losses import normalized_mae


class Learner:
    """Wraps a model and pipeline with fastai-style fit methods."""

    def __init__(
        self,
        model: nnx.Module,
        pipeline: GrainPipeline,
        loss_func=normalized_mae,
        n_skip: int = 0,
        metrics: list = [],
        plot_batch_fn=None,
        plot_results_fn=None,
    ):
        self.model = model
        self.pipeline = pipeline
        self.loss_func = loss_func
        self.n_skip = n_skip
        self.metrics = metrics
        self.plot_batch_fn = plot_batch_fn
        self.plot_results_fn = plot_results_fn
        self.train_losses: list[float] = []
        self.valid_losses: list[float] = []
        self.valid_metrics: dict[str, list[float]] = {m.__name__: [] for m in metrics}

    def _get_ds_and_source(self, split: str):
        """Return (iterable, source) for the given split name."""
        match split:
            case "train":
                return self.pipeline.train, self.pipeline.train_source
            case "valid":
                return self.pipeline.valid, self.pipeline.valid_source
            case "test":
                return self.pipeline.test, self.pipeline.test_source
            case _:
                raise ValueError(f"Unknown split: {split!r}. Use 'train', 'valid', or 'test'.")

    def show_batch(self, n: int = 4, split: str = "valid", figsize=None):
        """Plot input/output signals from a batch.

        Parameters
        ----------
        n : Number of samples to display.
        split : One of "train", "valid", "test".
        figsize : Matplotlib figure size override.
        """
        ds, source = self._get_ds_and_source(split)
        batch = next(iter(ds))

        if self.plot_batch_fn is not None:
            return self.plot_batch_fn(
                batch, n=n, figsize=figsize, source=source, pipeline=self.pipeline
            )

        from tsjax.viz import plot_batch

        input_key = self.pipeline.input_keys[0]
        target_key = self.pipeline.target_keys[0]
        return plot_batch(
            batch,
            n=n,
            figsize=figsize,
            u_labels=source.signal_names.get(input_key, [input_key]),
            y_labels=source.signal_names.get(target_key, [target_key]),
        )

    def show_results(self, n: int = 4, split: str = "valid", figsize=None):
        """Plot model predictions vs actual outputs.

        Parameters
        ----------
        n : Number of samples to display.
        split : One of "train", "valid", "test".
        figsize : Matplotlib figure size override.
        """
        ds, source = self._get_ds_and_source(split)
        batch = next(iter(ds))
        target_key = self.pipeline.target_keys[0]
        inputs = {k: jnp.asarray(batch[k]) for k in self.pipeline.input_keys}
        pred = np.asarray(self.model(**inputs))
        y = np.asarray(batch[target_key])
        if self.n_skip > 0:
            pred = pred[:, self.n_skip :]
            y = y[:, self.n_skip :]

        if self.plot_results_fn is not None:
            return self.plot_results_fn(
                target=y,
                pred=pred,
                n=n,
                figsize=figsize,
                batch=batch,
                source=source,
                pipeline=self.pipeline,
            )

        from tsjax.viz import plot_results

        input_key = self.pipeline.input_keys[0]
        return plot_results(
            target=y,
            pred=pred,
            n=n,
            figsize=figsize,
            u=np.asarray(batch[input_key]),
            y_labels=source.signal_names.get(target_key, [target_key]),
            u_labels=source.signal_names.get(input_key, [input_key]),
        )

    def fit(self, n_epoch: int, lr: float = 3e-3, progress: bool = True):
        """Train with constant LR."""
        tx = optax.adam(lr)
        self._fit(n_epoch, tx, progress=progress)

    def fit_flat_cos(
        self, n_epoch: int, lr: float = 3e-3, pct_start: float = 0.75, progress: bool = True
    ):
        """Train with flat LR then cosine decay."""
        batches_per_epoch = self.pipeline.n_train_batches
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
        target_key = self.pipeline.target_keys[0]
        target_stats = self.pipeline.stats[target_key]
        y_mean = jnp.asarray(target_stats.mean)
        y_std = jnp.asarray(target_stats.std)
        input_keys = self.pipeline.input_keys
        n_skip = self.n_skip
        loss_func = self.loss_func
        metric_fns = list(self.metrics)
        model = self.model

        optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)

        def loss_fn(model, inputs, y):
            pred = model(**inputs)
            if n_skip > 0:
                pred = pred[:, n_skip:]
                y = y[:, n_skip:]
            return loss_func(pred, y, y_mean, y_std)

        @nnx.jit
        def train_step(model, optimizer, inputs, y):
            loss, grads = nnx.value_and_grad(loss_fn)(model, inputs, y)
            optimizer.update(model, grads)
            return loss

        @nnx.jit
        def eval_step(model, inputs, y):
            pred = model(**inputs)
            if n_skip > 0:
                pred = pred[:, n_skip:]
                y = y[:, n_skip:]
            loss = loss_func(pred, y, y_mean, y_std)
            mvals = [m(pred, y, y_mean, y_std) for m in metric_fns]
            return loss, mvals

        n_total_batches = self.pipeline.n_train_batches
        train_iter = iter(self.pipeline.train)

        for epoch in range(n_epoch):
            epoch_start = time.time()
            # Training
            epoch_loss = 0.0
            pbar = None
            if progress:
                pbar = tqdm(
                    total=n_total_batches,
                    desc=f"Epoch {epoch + 1}/{n_epoch}",
                    leave=False,
                    mininterval=1.0,
                )
            for i in range(n_total_batches):
                batch = next(train_iter)
                inputs = {k: jnp.asarray(batch[k]) for k in input_keys}
                y = jnp.asarray(batch[target_key])
                loss = train_step(model, optimizer, inputs, y)
                epoch_loss += float(loss)
                if pbar is not None:
                    pbar.update(1)
                    pbar.set_postfix(loss=f"{epoch_loss / (i + 1):.4f}", refresh=False)
            if pbar is not None:
                pbar.close()

            avg_train = epoch_loss / max(n_total_batches, 1)
            self.train_losses.append(avg_train)

            # Validation
            val_loss = 0.0
            metric_sums = [0.0] * len(metric_fns)
            n_val = 0
            for batch in self.pipeline.valid:
                inputs = {k: jnp.asarray(batch[k]) for k in input_keys}
                y = jnp.asarray(batch[target_key])
                loss, mvals = eval_step(model, inputs, y)
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
