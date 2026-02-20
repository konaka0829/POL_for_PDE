"""PDE Operator Learning (POL) modules for backprop-free reservoir methods."""

from .elm import FixedRandomELM
from .features_1d import (
    build_sensor_indices,
    build_time_grid,
    collect_observations,
    flatten_observations,
)
from .reservoir_1d import Reservoir1DSolver, ReservoirConfig
from .ridge import fit_ridge_streaming, predict_linear
from .theme1_random_features_1d import (
    RandomReservoirFeatureMap1D,
    sample_reservoir_configs,
)

__all__ = [
    "FixedRandomELM",
    "Reservoir1DSolver",
    "ReservoirConfig",
    "build_time_grid",
    "build_sensor_indices",
    "collect_observations",
    "flatten_observations",
    "fit_ridge_streaming",
    "predict_linear",
    "sample_reservoir_configs",
    "RandomReservoirFeatureMap1D",
]
