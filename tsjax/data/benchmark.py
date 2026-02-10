"""IdentiBench integration — create pipelines from benchmark specs.

All imports from :mod:`.pipeline` (which pulls in ``grain``) are deferred to
function bodies so that this module can be imported alongside ``identibench``
(which pulls in ``sklearn``) without triggering the grain/sklearn mutex
conflict.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from identibench.benchmark import BenchmarkSpecBase

    from .pipeline import GrainPipeline

# Per-benchmark dataloader defaults, matching TSFast.
BENCHMARK_DL_KWARGS: dict[str, dict[str, Any]] = {
    # Simulation benchmarks
    "BenchmarkWH_Simulation": {"win_sz": 200},
    "BenchmarkSilverbox_Simulation": {"win_sz": 200},
    "BenchmarkCascadedTanks_Simulation": {"win_sz": 150, "bs": 16},
    "BenchmarkEMPS_Simulation": {"win_sz": 1000},
    "BenchmarkCED_Simulation": {"win_sz": 100, "bs": 16},
    "BenchmarkNoisyWH_Simulation": {"win_sz": 100, "stp_sz": 50},
    "BenchmarkRobotForward_Simulation": {"win_sz": 300, "valid_stp_sz": 4},
    "BenchmarkRobotInverse_Simulation": {"win_sz": 300, "valid_stp_sz": 4},
    "BenchmarkShip_Simulation": {"win_sz": 100},
    "BenchmarkQuadPelican_Simulation": {"win_sz": 300, "valid_stp_sz": 40},
    "BenchmarkQuadPi_Simulation": {"win_sz": 200, "valid_stp_sz": 20},
    # Prediction benchmarks
    "BenchmarkWH_Prediction": {},
    "BenchmarkSilverbox_Prediction": {},
    "BenchmarkCascadedTanks_Prediction": {"bs": 16},
    "BenchmarkEMPS_Prediction": {},
    "BenchmarkCED_Prediction": {"bs": 16},
    "BenchmarkNoisyWH_Prediction": {"stp_sz": 50},
    "BenchmarkRobotForward_Prediction": {"valid_stp_sz": 4},
    "BenchmarkRobotInverse_Prediction": {"valid_stp_sz": 4},
    "BenchmarkShip_Prediction": {},
    "BenchmarkQuadPelican_Prediction": {"valid_stp_sz": 40},
    "BenchmarkQuadPi_Prediction": {"valid_stp_sz": 20},
}

# Valid keyword arguments for create_grain_dls (excluding positional inputs, targets, dataset).
# Hardcoded to avoid importing .pipeline (and thus grain) at module level.
_VALID_DL_KWARGS = {
    "win_sz",
    "stp_sz",
    "valid_stp_sz",
    "bs",
    "seed",
    "preload",
    "resampling_factor",
    "target_fs",
    "fs_attr",
    "resample_fn",
}


def _build_pipeline_kwargs(spec: BenchmarkSpecBase, **kwargs: Any) -> dict[str, Any]:
    """Build merged kwargs for ``create_grain_dls`` from a benchmark spec.

    Merge precedence (last wins):
    spec fields -> prediction window sizing -> BENCHMARK_DL_KWARGS -> user kwargs
    """
    import identibench.benchmark as idb_bench

    merged: dict[str, Any] = {
        "inputs": {"u": spec.u_cols},
        "targets": {"y": spec.y_cols},
        "dataset": spec.dataset_path,
    }

    # Prediction specs: auto-compute window sizing
    if isinstance(spec, idb_bench.BenchmarkSpecPrediction):
        merged["win_sz"] = spec.pred_horizon + spec.init_window
        merged["valid_stp_sz"] = spec.pred_step

    # Benchmark-specific defaults
    if spec.name in BENCHMARK_DL_KWARGS:
        merged.update(BENCHMARK_DL_KWARGS[spec.name])

    # User kwargs win
    merged.update(kwargs)
    return merged


def create_grain_dls_from_spec(spec: BenchmarkSpecBase, **kwargs: Any) -> GrainPipeline:
    """Create a :class:`GrainPipeline` from an IdentiBench benchmark spec.

    Downloads the dataset if it hasn't been fetched yet, merges benchmark-
    specific defaults with any user overrides, and delegates to
    :func:`create_grain_dls`.

    Parameters
    ----------
    spec : identibench.benchmark.BenchmarkSpecBase
        An IdentiBench benchmark specification (simulation or prediction).
    **kwargs
        Overrides forwarded to :func:`create_grain_dls` (e.g. ``bs``,
        ``win_sz``, ``stp_sz``).
    """
    from .pipeline import create_grain_dls

    spec.ensure_dataset_exists()
    merged = _build_pipeline_kwargs(spec, **kwargs)
    return create_grain_dls(**merged)
