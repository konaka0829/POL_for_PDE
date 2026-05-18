"""PDE Operator Learning (POL) modules for backprop-free reservoir methods."""

from .elm import FixedRandomELM
from .burgers_spectral_1d import (
    burgers_nonlinear_hat,
    burgers_split_step_outer,
    make_dealias_mask,
    make_wavenumbers,
    simulate_burgers_split_step,
)
from .features_1d import (
    build_sensor_indices,
    build_time_grid,
    collect_observations,
    flatten_observations,
)
from .time_grid import require_time_aligned
from .reservoir_1d import Reservoir1DSolver
from .ridge import fit_ridge_streaming, fit_ridge_streaming_standardized, predict_linear
from .model123_1d import DatasetBundle, DatasetConfig, ExperimentConfig, build_dataset, run_experiment, save_dataset_bundle

__all__ = [
    "FixedRandomELM",
    "make_wavenumbers",
    "make_dealias_mask",
    "burgers_nonlinear_hat",
    "burgers_split_step_outer",
    "simulate_burgers_split_step",
    "Reservoir1DSolver",
    "build_time_grid",
    "require_time_aligned",
    "build_sensor_indices",
    "collect_observations",
    "flatten_observations",
    "fit_ridge_streaming",
    "fit_ridge_streaming_standardized",
    "predict_linear",
    "DatasetBundle",
    "DatasetConfig",
    "ExperimentConfig",
    "build_dataset",
    "run_experiment",
    "save_dataset_bundle",
]
