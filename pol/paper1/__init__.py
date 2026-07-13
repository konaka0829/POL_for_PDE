"""Phase 1 components for the first paper implementation."""

from .config import (
    DataConfig,
    DomainConfig,
    Paper1Config,
    SpatialConfig,
    TargetPDEConfig,
    load_config_json,
    save_config_json,
)

__all__ = [
    "DataConfig",
    "DomainConfig",
    "Paper1Config",
    "SpatialConfig",
    "TargetPDEConfig",
    "load_config_json",
    "save_config_json",
]
