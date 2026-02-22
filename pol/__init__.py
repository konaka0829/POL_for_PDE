"""PDE Operator Learning (POL) modules for backprop-free reservoir methods."""

from .elm import FixedRandomELM
from .encoder_1d import EncoderOutputs, FixedEncoder1D
from .features_1d import (
    build_sensor_indices,
    build_time_grid,
    collect_observations,
    flatten_observations,
)
from .reservoir_1d import Reservoir1DSolver
from .ridge import fit_ridge_streaming, fit_ridge_streaming_standardized, predict_linear

__all__ = [
    "FixedRandomELM",
    "FixedEncoder1D",
    "EncoderOutputs",
    "Reservoir1DSolver",
    "build_time_grid",
    "build_sensor_indices",
    "collect_observations",
    "flatten_observations",
    "fit_ridge_streaming",
    "fit_ridge_streaming_standardized",
    "predict_linear",
]
