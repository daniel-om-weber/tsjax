"""Normalization layers for composable model wrapping."""

from __future__ import annotations

import jax.numpy as jnp
from flax import nnx

from tsjax._core import Buffer


class Normalize(nnx.Module):
    """Input normalization: ``(x - mean) / std``.

    Stores mean/std as non-trainable :class:`Buffer` variables.
    Defaults to identity (mean=0, std=1) when stats are not provided.
    """

    def __init__(self, size: int, mean=None, std=None):
        self.mean = Buffer(jnp.zeros(size) if mean is None else jnp.asarray(mean))
        self.std = Buffer(jnp.ones(size) if std is None else jnp.asarray(std))

    def __call__(self, x):
        return (x - self.mean[...]) / self.std[...]


class Denormalize(nnx.Module):
    """Output denormalization: ``x * std + mean``.

    Stores mean/std as non-trainable :class:`Buffer` variables.
    Defaults to identity (mean=0, std=1) when stats are not provided.
    """

    def __init__(self, size: int, mean=None, std=None):
        self.mean = Buffer(jnp.zeros(size) if mean is None else jnp.asarray(mean))
        self.std = Buffer(jnp.ones(size) if std is None else jnp.asarray(std))

    def __call__(self, x):
        return x * self.std[...] + self.mean[...]


class NormalizedModel(nnx.Module):
    """Wrap any model with input normalization and output denormalization.

    Follows the ``nnx.WeightNorm`` / ``nnx.SpectralNorm`` wrapper pattern.
    Raw physical values in, raw physical values out.

    Alternative: use ``nnx.Sequential(Normalize(...), model, Denormalize(...))``
    for the same effect without named sub-module access.
    """

    def __init__(
        self,
        model: nnx.Module,
        norm_in: Normalize,
        norm_out: Denormalize,
    ):
        self.norm_in = norm_in
        self.model = model
        self.norm_out = norm_out

    def __call__(self, *args, **kwargs):
        if args:
            first, *rest = args
            x = self.norm_in(first)
            x = self.model(x, *rest, **kwargs)
        else:
            # kwargs-only call (e.g. model(u=tensor)) — normalize first kwarg
            first_key = next(iter(kwargs))
            kwargs = {**kwargs, first_key: self.norm_in(kwargs[first_key])}
            x = self.model(**kwargs)
        return self.norm_out(x)
