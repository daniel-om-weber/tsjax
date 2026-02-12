"""Protocol for format-agnostic signal file stores."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class SignalStore(Protocol):
    """Read-only store over signal files.

    Any object satisfying this protocol can be used with WindowedSource,
    FileSource, and SignalReader.  Implementations must be picklable
    (Grain multiprocessing requirement).
    """

    @property
    def paths(self) -> Sequence[str]:
        """Ordered file paths known to this store."""
        ...

    def get_seq_len(self, path: str, signal: str | None = None) -> int:
        """Return the sequence length for *path* (first signal when None)."""
        ...

    def read_signals(self, path: str, signals: list[str], l_slc: int, r_slc: int) -> np.ndarray:
        """Read and stack *signals* into shape (r_slc - l_slc, len(signals))."""
        ...
