"""Shared fixtures for tsjax integration tests."""

import matplotlib

matplotlib.use("Agg")

from pathlib import Path

import pytest

DATASET = Path(__file__).parent.parent / "test_data" / "WienerHammerstein"
U_SIGNALS = ["u"]
Y_SIGNALS = ["y"]
WIN_SZ = 20
STP_SZ = 10
BS = 4


@pytest.fixture(scope="session")
def dataset_path():
    return DATASET


@pytest.fixture(scope="session")
def pipeline():
    """Session-scoped pipeline — expensive to create, shared across all tests."""
    from tsjax import create_simulation_dls

    return create_simulation_dls(
        u=U_SIGNALS,
        y=Y_SIGNALS,
        dataset=DATASET,
        win_sz=WIN_SZ,
        stp_sz=STP_SZ,
        bs=BS,
        seed=42,
        preload=True,
    )
