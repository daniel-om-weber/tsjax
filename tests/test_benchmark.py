"""Tests for IdentiBench integration (tsjax.data.benchmark).

These tests run in a subprocess to avoid the grain/sklearn mutex conflict.
identibench imports sklearn at module level; grain (used by the pipeline
fixture and other tests) deadlocks if sklearn is already loaded.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

# Skip the entire module if identibench is not installed, without importing it.
try:
    from importlib.util import find_spec

    _has_idb = find_spec("identibench") is not None
except ImportError:
    _has_idb = False

pytestmark = pytest.mark.skipif(not _has_idb, reason="identibench not installed")


def _run_in_subprocess(code: str) -> None:
    """Run *code* in a fresh Python process and assert success."""
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        pytest.fail(f"Subprocess failed:\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}")


# ---------------------------------------------------------------------------
# Unit tests — _build_pipeline_kwargs
# ---------------------------------------------------------------------------


class TestBuildPipelineKwargs:
    """Verify merge logic without downloading data."""

    def test_spec_fields_extracted(self):
        _run_in_subprocess("""\
            import identibench as idb
            from tsjax.data.benchmark import _build_pipeline_kwargs
            spec = idb.BenchmarkWH_Simulation
            kw = _build_pipeline_kwargs(spec)
            assert kw["inputs"] == {"u": spec.u_cols}
            assert kw["targets"] == {"y": spec.y_cols}
            assert kw["dataset"] == spec.dataset_path
        """)

    def test_simulation_uses_benchmark_defaults(self):
        _run_in_subprocess("""\
            import identibench as idb
            from tsjax.data.benchmark import BENCHMARK_DL_KWARGS, _build_pipeline_kwargs
            spec = idb.BenchmarkWH_Simulation
            kw = _build_pipeline_kwargs(spec)
            expected = BENCHMARK_DL_KWARGS[spec.name]
            for k, v in expected.items():
                assert kw[k] == v, f"{k}: expected {v}, got {kw[k]}"
        """)

    def test_prediction_computes_win_sz(self):
        _run_in_subprocess("""\
            import identibench as idb
            from tsjax.data.benchmark import _build_pipeline_kwargs
            spec = idb.BenchmarkWH_Prediction
            kw = _build_pipeline_kwargs(spec)
            assert kw["win_sz"] == spec.pred_horizon + spec.init_window
        """)

    def test_prediction_sets_valid_stp_sz(self):
        _run_in_subprocess("""\
            import identibench as idb
            from tsjax.data.benchmark import _build_pipeline_kwargs
            spec = idb.BenchmarkWH_Prediction
            kw = _build_pipeline_kwargs(spec)
            assert kw["valid_stp_sz"] == spec.pred_step
        """)

    def test_benchmark_defaults_override_prediction_sizing(self):
        _run_in_subprocess("""\
            import identibench as idb
            from tsjax.data.benchmark import _build_pipeline_kwargs
            spec = idb.BenchmarkCascadedTanks_Prediction
            kw = _build_pipeline_kwargs(spec)
            assert kw["bs"] == 16
        """)

    def test_user_kwargs_override_everything(self):
        _run_in_subprocess("""\
            import identibench as idb
            from tsjax.data.benchmark import _build_pipeline_kwargs
            spec = idb.BenchmarkWH_Simulation
            kw = _build_pipeline_kwargs(spec, win_sz=999, bs=1)
            assert kw["win_sz"] == 999
            assert kw["bs"] == 1
        """)

    def test_user_kwargs_override_prediction_win_sz(self):
        _run_in_subprocess("""\
            import identibench as idb
            from tsjax.data.benchmark import _build_pipeline_kwargs
            spec = idb.BenchmarkWH_Prediction
            kw = _build_pipeline_kwargs(spec, win_sz=42)
            assert kw["win_sz"] == 42
        """)


# ---------------------------------------------------------------------------
# Structural tests
# ---------------------------------------------------------------------------


class TestBenchmarkCoverage:
    def test_all_simulation_benchmarks_have_entries(self):
        _run_in_subprocess("""\
            import identibench as idb
            from tsjax.data.benchmark import BENCHMARK_DL_KWARGS
            for spec in idb.simulation_benchmarks.values():
                assert spec.name in BENCHMARK_DL_KWARGS, f"Missing entry for {spec.name}"
        """)

    def test_all_prediction_benchmarks_have_entries(self):
        _run_in_subprocess("""\
            import identibench as idb
            from tsjax.data.benchmark import BENCHMARK_DL_KWARGS
            for spec in idb.prediction_benchmarks.values():
                assert spec.name in BENCHMARK_DL_KWARGS, f"Missing entry for {spec.name}"
        """)

    def test_no_extra_entries(self):
        _run_in_subprocess("""\
            import identibench as idb
            from tsjax.data.benchmark import BENCHMARK_DL_KWARGS
            known = {s.name for s in idb.simulation_benchmarks.values()} | {
                s.name for s in idb.prediction_benchmarks.values()
            }
            for name in BENCHMARK_DL_KWARGS:
                assert name in known, f"Unknown benchmark in BENCHMARK_DL_KWARGS: {name}"
        """)


class TestKwargsValidity:
    def test_all_kwargs_keys_are_valid(self):
        _run_in_subprocess("""\
            from tsjax.data.benchmark import _VALID_DL_KWARGS, BENCHMARK_DL_KWARGS
            for name, kw in BENCHMARK_DL_KWARGS.items():
                for k in kw:
                    assert k in _VALID_DL_KWARGS, (
                        f"Invalid kwarg '{k}' in BENCHMARK_DL_KWARGS['{name}']; "
                        f"valid: {sorted(_VALID_DL_KWARGS)}"
                    )
        """)


# ---------------------------------------------------------------------------
# Integration-level: create_grain_dls_from_spec delegates correctly
# ---------------------------------------------------------------------------


class TestCreateGrainDlsFromSpec:
    def test_calls_ensure_and_delegates(self):
        _run_in_subprocess("""\
            from unittest.mock import patch
            import identibench as idb
            spec = idb.BenchmarkWH_Simulation
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
            assert call_kw.kwargs["inputs"] == {"u": spec.u_cols}
            assert call_kw.kwargs["targets"] == {"y": spec.y_cols}
            assert result is sentinel
        """)
