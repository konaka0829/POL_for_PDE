"""Reservoir-FNO components for Darcy 2D experiments."""

from .models_fno2d import FNO2d
from .reservoir_readout import RidgeReadout2D
from .darcy_generate import generate_darcy_dataset, save_dataset_to_mat
from .data_utils import MatReader, UnitGaussianNormalizer

__all__ = [
    "FNO2d",
    "RidgeReadout2D",
    "generate_darcy_dataset",
    "save_dataset_to_mat",
    "MatReader",
    "UnitGaussianNormalizer",
]
