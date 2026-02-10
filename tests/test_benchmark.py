"""Tests for IdentiBench integration (tsjax.data.benchmark)."""

# ruff: noqa: E402 — imports must follow pytest.importorskip
from __future__ import annotations

from unittest.mock import patch

import pytest

idb = pytest.importorskip("identibench")

from tsjax.data.benchmark import (
    _VALID_DL_KWARGS,
    BENCHMARK_DL_KWARGS,
    _build_pipeline_kwargs,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sim_spec():
    """Return a representative simulation spec."""
    return idb.BenchmarkWH_Simulation


def _pred_spec():
    """Return a representative prediction spec."""
    return idb.BenchmarkWH_Prediction


# ---------------------------------------------------------------------------
# Unit tests — _build_pipeline_kwargs
# ---------------------------------------------------------------------------

class TestBuildPipelineKwargs:
    """Verify merge logic without downloading data."""

    def test_spec_fields_extracted(self):
        spec = _sim_spec()
        kw = _build_pipeline_kwargs(spec)
        assert kw["u"] == spec.u_cols
        assert kw["y"] == spec.y_cols
        assert kw["dataset"] == spec.dataset_path

    def test_simulation_uses_benchmark_defaults(self):
        spec = _sim_spec()
        kw = _build_pipeline_kwargs(spec)
        expected = BENCHMARK_DL_KWARGS[spec.name]
        for k, v in expected.items():
            assert kw[k] == v, f"{k}: expected {v}, got {kw[k]}"

    def test_prediction_computes_win_sz(self):
        spec = _pred_spec()
        kw = _build_pipeline_kwargs(spec)
        assert kw["win_sz"] == spec.pred_horizon + spec.init_window

    def test_prediction_sets_valid_stp_sz(self):
        spec = _pred_spec()
        kw = _build_pipeline_kwargs(spec)
        assert kw["valid_stp_sz"] == spec.pred_step

    def test_benchmark_defaults_override_prediction_sizing(self):
        """BENCHMARK_DL_KWARGS entries override prediction-computed values."""
        spec = idb.BenchmarkCascadedTanks_Prediction
        kw = _build_pipeline_kwargs(spec)
        # CascadedTanks_Prediction has bs=16 in BENCHMARK_DL_KWARGS
        assert kw["bs"] == 16

    def test_user_kwargs_override_everything(self):
        spec = _sim_spec()
        kw = _build_pipeline_kwargs(spec, win_sz=999, bs=1)
        assert kw["win_sz"] == 999
        assert kw["bs"] == 1

    def test_user_kwargs_override_prediction_win_sz(self):
        spec = _pred_spec()
        kw = _build_pipeline_kwargs(spec, win_sz=42)
        assert kw["win_sz"] == 42


# ---------------------------------------------------------------------------
# Structural tests
# ---------------------------------------------------------------------------

class TestBenchmarkCoverage:
    """Ensure BENCHMARK_DL_KWARGS covers all IdentiBench benchmarks."""

    def test_all_simulation_benchmarks_have_entries(self):
        for spec in idb.simulation_benchmarks.values():
            assert spec.name in BENCHMARK_DL_KWARGS, f"Missing entry for {spec.name}"

    def test_all_prediction_benchmarks_have_entries(self):
        for spec in idb.prediction_benchmarks.values():
            assert spec.name in BENCHMARK_DL_KWARGS, f"Missing entry for {spec.name}"

    def test_no_extra_entries(self):
        """BENCHMARK_DL_KWARGS should not contain stale/unknown keys."""
        known = {s.name for s in idb.simulation_benchmarks.values()} | {
            s.name for s in idb.prediction_benchmarks.values()
        }
        for name in BENCHMARK_DL_KWARGS:
            assert name in known, f"Unknown benchmark in BENCHMARK_DL_KWARGS: {name}"


class TestKwargsValidity:
    """Ensure all values in BENCHMARK_DL_KWARGS are valid create_grain_dls params."""

    def test_all_kwargs_keys_are_valid(self):
        for name, kw in BENCHMARK_DL_KWARGS.items():
            for k in kw:
                assert k in _VALID_DL_KWARGS, (
                    f"Invalid kwarg '{k}' in BENCHMARK_DL_KWARGS['{name}']; "
                    f"valid: {sorted(_VALID_DL_KWARGS)}"
                )


# ---------------------------------------------------------------------------
# Integration-level: create_grain_dls_from_spec delegates correctly
# ---------------------------------------------------------------------------

class TestCreateGrainDlsFromSpec:
    """Test that create_grain_dls_from_spec wires everything together."""

    def test_calls_ensure_and_delegates(self):
        """Verify ensure_dataset_exists is called and create_grain_dls receives merged kwargs."""
        spec = _sim_spec()
        sentinel = object()

        with (
            patch.object(spec, "ensure_dataset_exists") as mock_ensure,
            patch(
                "tsjax.data.pipeline.create_grain_dls", return_value=sentinel
            ) as mock_create,
        ):
            from tsjax.data.benchmark import create_grain_dls_from_spec

            result = create_grain_dls_from_spec(spec, bs=7)

        mock_ensure.assert_called_once()
        mock_create.assert_called_once()
        call_kw = mock_create.call_args
        assert call_kw.kwargs["bs"] == 7
        assert call_kw.kwargs["u"] == spec.u_cols
        assert result is sentinel
