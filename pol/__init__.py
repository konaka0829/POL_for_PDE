"""PDE Operator Learning (POL) with lazy legacy compatibility exports."""
from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORTS = {
    "FixedRandomELM": (".elm", "FixedRandomELM"),
    "make_wavenumbers": (".burgers_spectral_1d", "make_wavenumbers"),
    "make_dealias_mask": (".burgers_spectral_1d", "make_dealias_mask"),
    "burgers_nonlinear_hat": (".burgers_spectral_1d", "burgers_nonlinear_hat"),
    "burgers_split_step_outer": (
        ".burgers_spectral_1d",
        "burgers_split_step_outer",
    ),
    "simulate_burgers_split_step": (
        ".burgers_spectral_1d",
        "simulate_burgers_split_step",
    ),
    "Reservoir1DSolver": (".reservoir_1d", "Reservoir1DSolver"),
    "build_time_grid": (".features_1d", "build_time_grid"),
    "require_time_aligned": (".time_grid", "require_time_aligned"),
    "build_sensor_indices": (".features_1d", "build_sensor_indices"),
    "collect_observations": (".features_1d", "collect_observations"),
    "flatten_observations": (".features_1d", "flatten_observations"),
    "fit_ridge_streaming": (".ridge", "fit_ridge_streaming"),
    "fit_ridge_streaming_standardized": (
        ".ridge",
        "fit_ridge_streaming_standardized",
    ),
    "predict_linear": (".ridge", "predict_linear"),
    "DatasetBundle": (".model123_1d", "DatasetBundle"),
    "DatasetConfig": (".model123_1d", "DatasetConfig"),
    "ExperimentConfig": (".model123_1d", "ExperimentConfig"),
    "build_dataset": (".model123_1d", "build_dataset"),
    "run_experiment": (".model123_1d", "run_experiment"),
    "save_dataset_bundle": (".model123_1d", "save_dataset_bundle"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    """Load a legacy public export only when it is first requested."""
    try:
        module_name, attribute_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name, __name__), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Return module globals plus lazy public exports."""
    return sorted(set(globals()) | set(__all__))
