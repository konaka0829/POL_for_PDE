from __future__ import annotations

from dataclasses import asdict, dataclass, fields, is_dataclass
import json
from pathlib import Path
from typing import Any

import torch


_DTYPES = {"float32": torch.float32, "float64": torch.float64}
_DEVICES = {"cpu", "cuda", "auto"}
_SOLVERS = {"split_step", "semi_implicit", "etdrk4", "fourier_pseudospectral_etdrk4"}


@dataclass(frozen=True)
class DomainConfig:
    length: float = 1.0


@dataclass(frozen=True)
class DataConfig:
    total_samples: int
    n_train: int
    n_val: int
    n_test: int
    seed: int = 0
    dtype: str = "float64"
    device: str = "cpu"
    ic_type: str = "grf"
    grf_gamma: float = 2.0
    grf_tau: float = 5.0
    grf_sigma: float = 25.0
    grf_mean: float = 0.0
    preprocessing: str = "l2_scaling_only"

    def torch_dtype(self) -> torch.dtype:
        return _DTYPES[self.dtype]


@dataclass(frozen=True)
class TargetPDEConfig:
    equation: str = "viscous_burgers"
    nu: float = 1e-2
    T: float = 1.0
    dt: float = 1e-3
    fine_dt: float = 1e-4
    solver: str = "split_step"
    dealias: bool = True


@dataclass(frozen=True)
class SpatialConfig:
    target_master_nx: int
    target_data_nx: int
    surrogate_internal_nx: int
    observation_dim: int
    target_output_dim: int


@dataclass(frozen=True)
class Paper1Config:
    domain: DomainConfig
    data: DataConfig
    target: TargetPDEConfig
    spatial: SpatialConfig

    def validate(self) -> "Paper1Config":
        if self.domain.length <= 0.0:
            raise ValueError("domain.length must be positive")
        if self.data.total_samples != self.data.n_train + self.data.n_val + self.data.n_test:
            raise ValueError("data.total_samples must equal n_train + n_val + n_test")
        if self.data.n_train <= 0:
            raise ValueError("data.n_train must be positive")
        if self.data.n_val < 0 or self.data.n_test < 0:
            raise ValueError("data.n_val and data.n_test must be nonnegative")
        if self.data.total_samples <= 0:
            raise ValueError("data.total_samples must be positive")
        if self.data.dtype not in _DTYPES:
            raise ValueError(f"unsupported data.dtype: {self.data.dtype}")
        if self.data.device not in _DEVICES:
            raise ValueError(f"unsupported data.device: {self.data.device}")
        if self.data.ic_type != "grf":
            raise ValueError("Phase 1 supports data.ic_type='grf'")
        if self.data.grf_gamma <= 0.0:
            raise ValueError("data.grf_gamma must be positive")
        if self.data.grf_tau < 0.0 or self.data.grf_sigma < 0.0:
            raise ValueError("data.grf_tau and data.grf_sigma must be nonnegative")
        if "z" in self.data.preprocessing.lower() and "score" in self.data.preprocessing.lower():
            raise ValueError("component-wise z-score preprocessing is not enabled for Phase 1")

        if self.target.equation not in {"burgers", "viscous_burgers"}:
            raise ValueError("target.equation must be viscous_burgers/burgers")
        if self.target.nu <= 0.0:
            raise ValueError("target.nu must be positive")
        if self.target.T <= 0.0 or self.target.dt <= 0.0 or self.target.fine_dt <= 0.0:
            raise ValueError("target.T, target.dt, and target.fine_dt must be positive")
        if self.target.solver not in _SOLVERS:
            raise ValueError(f"unsupported target.solver: {self.target.solver}")

        dims = self.spatial
        for name in ("target_master_nx", "target_data_nx", "surrogate_internal_nx", "observation_dim", "target_output_dim"):
            if getattr(dims, name) <= 0:
                raise ValueError(f"spatial.{name} must be positive")
        if dims.target_data_nx > dims.target_master_nx:
            raise ValueError("spatial.target_data_nx must be <= spatial.target_master_nx")
        if dims.observation_dim > dims.surrogate_internal_nx:
            raise ValueError("spatial.observation_dim must be <= spatial.surrogate_internal_nx")
        q = dims.target_output_dim
        if q % 2 == 0:
            raise ValueError("spatial.target_output_dim must be odd")
        kmax = (q - 1) // 2
        if kmax >= dims.target_data_nx / 2:
            raise ValueError("spatial.target_output_dim Fourier band is not representable on target_data_nx")
        return self

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _filter_dataclass(cls, values: dict[str, Any]):
    names = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in values.items() if k in names})


def config_from_dict(raw: dict[str, Any]) -> Paper1Config:
    cfg = Paper1Config(
        domain=_filter_dataclass(DomainConfig, raw.get("domain", {})),
        data=_filter_dataclass(DataConfig, raw.get("data", {})),
        target=_filter_dataclass(TargetPDEConfig, raw.get("target", {})),
        spatial=_filter_dataclass(SpatialConfig, raw.get("spatial", {})),
    )
    return cfg.validate()


def load_config_json(path: str | Path) -> Paper1Config:
    with Path(path).open("r", encoding="utf-8") as f:
        return config_from_dict(json.load(f))


def _canonical(obj: Any) -> Any:
    if is_dataclass(obj):
        return {k: _canonical(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {str(k): _canonical(v) for k, v in sorted(obj.items())}
    if isinstance(obj, (list, tuple)):
        return [_canonical(v) for v in obj]
    return obj


def canonical_config_json(config: Paper1Config) -> str:
    return json.dumps(_canonical(config), sort_keys=True, separators=(",", ":"))


def save_config_json(config: Paper1Config, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(config.to_dict(), f, indent=2, sort_keys=True)
        f.write("\n")
