"""Grain-compatible transforms for the data pipeline."""
from __future__ import annotations

import numpy as np


class NormalizeInputTransform:
    """Normalize only the 'u' (input) key, leave 'y' (output) untouched.

    Replaces fastai's type-dispatched Normalize that only fires on TensorSequencesInput.
    """

    def __init__(self, mean: np.ndarray, std: np.ndarray):
        self.mean = mean.astype(np.float32)
        self.std = std.astype(np.float32)

    def __call__(self, item: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        return {
            'u': (item['u'] - self.mean) / self.std,
            'y': item['y'],
        }
