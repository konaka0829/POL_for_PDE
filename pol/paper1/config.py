from __future__ import annotations

from dataclasses import asdict, dataclass, fields, is_dataclass
import json
from pathlib import Path
from typing import Any

import torch

from .solvers import effective_inner_step, normalize_burgers_solver_name


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
    fine_dt: float | None = 1e-4
    solver: str = "split_step"
    dealias: bool = True


@dataclass(frozen=True)
class SpatialConfig:
    reference_nx: int
    target_data_nx: int
    surrogate_internal_nx: int
    observation_dim: int
    target_output_dim: int

    @property
    def target_master_nx(self) -> int:
        """Legacy read-only alias for :attr:`reference_nx`."""
        return self.reference_nx


@dataclass(frozen=True)
class E0TimeCandidateConfig:
    dt: float
    fine_dt: float | None = None


@dataclass(frozen=True)
class E0ReferenceTolerancesConfig:
    mean_relative_l2: float
    max_relative_l2: float
    low_mode_relative_l2: float


@dataclass(frozen=True)
class E0AlgebraicTolerancesConfig:
    float64_atol: float = 1e-10
    float64_rtol: float = 1e-10
    float32_atol: float = 1e-5
    float32_rtol: float = 1e-5


@dataclass(frozen=True)
class E0Model1IdentityConfig:
    target_data_nx: int
    surrogate_internal_nx: int
    observation_dim: int
    target_output_dim: int


@dataclass(frozen=True)
class E0ReducedJConfig:
    observation_dim: int
    target_output_dim: int


@dataclass(frozen=True)
class E0Config:
    calibration_sample_ids: tuple[int, ...]
    reference_nx_candidates: tuple[int, ...]
    time_candidates: tuple[E0TimeCandidateConfig, ...]
    q_reference_check: int
    reference_tolerances: E0ReferenceTolerancesConfig
    algebraic_tolerances: E0AlgebraicTolerancesConfig
    model1_identity: E0Model1IdentityConfig
    reduced_j: E0ReducedJConfig
    profile: str = "smoke"
    selection_policy: str = "coarsest_passing_with_finest_pair_required"


@dataclass(frozen=True)
class Paper1Config:
    domain: DomainConfig
    data: DataConfig
    target: TargetPDEConfig
    spatial: SpatialConfig
    e0: E0Config | None = None

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
        if self.target.T <= 0.0 or self.target.dt <= 0.0:
            raise ValueError("target.T and target.dt must be positive")
        if self.target.solver not in _SOLVERS:
            raise ValueError(f"unsupported target.solver: {self.target.solver}")
        normalized_solver = normalize_burgers_solver_name(self.target.solver)
        if normalized_solver == "split_step" and (self.target.fine_dt is None or self.target.fine_dt <= 0):
            raise ValueError("split-step target requires positive target.fine_dt")
        if normalized_solver == "etdrk4" and self.target.fine_dt is not None and self.target.fine_dt <= 0:
            raise ValueError("target.fine_dt must be positive when provided")

        dims = self.spatial
        for name in ("reference_nx", "target_data_nx", "surrogate_internal_nx", "observation_dim", "target_output_dim"):
            if getattr(dims, name) <= 0:
                raise ValueError(f"spatial.{name} must be positive")
        if dims.target_data_nx > dims.reference_nx:
            raise ValueError("spatial.target_data_nx must be <= spatial.reference_nx")
        if dims.observation_dim > dims.surrogate_internal_nx:
            raise ValueError("spatial.observation_dim must be <= spatial.surrogate_internal_nx")
        q = dims.target_output_dim
        if q % 2 == 0:
            raise ValueError("spatial.target_output_dim must be odd")
        kmax = (q - 1) // 2
        if kmax >= dims.target_data_nx / 2:
            raise ValueError("spatial.target_output_dim Fourier band is not representable on target_data_nx")
        if self.e0 is not None:
            e0 = self.e0
            if not e0.calibration_sample_ids:
                raise ValueError("e0.calibration_sample_ids must be non-empty")
            if len(set(e0.calibration_sample_ids)) != len(e0.calibration_sample_ids):
                raise ValueError("e0.calibration_sample_ids must not contain duplicates")
            if any(i < 0 or i >= self.data.total_samples for i in e0.calibration_sample_ids):
                raise ValueError("e0.calibration_sample_ids contains an invalid sample ID")
            if len(e0.reference_nx_candidates) < 2:
                raise ValueError("e0.reference_nx_candidates must contain at least two values")
            if tuple(sorted(set(e0.reference_nx_candidates))) != e0.reference_nx_candidates:
                raise ValueError("e0.reference_nx_candidates must be strictly increasing and unique")
            if e0.reference_nx_candidates[-1] != dims.reference_nx:
                raise ValueError("largest e0.reference_nx_candidates value must equal spatial.reference_nx")
            if len(e0.time_candidates) < 2:
                raise ValueError("e0.time_candidates must contain at least two values")
            if e0.selection_policy != "coarsest_passing_with_finest_pair_required":
                raise ValueError(f"unsupported e0.selection_policy: {e0.selection_policy}")
            effective_steps: list[float] = []
            for i, candidate in enumerate(e0.time_candidates):
                if candidate.dt <= 0 or (candidate.fine_dt is not None and candidate.fine_dt <= 0):
                    raise ValueError(f"e0.time_candidates[{i}] steps must be positive")
                outer = round(self.target.T / candidate.dt)
                if abs(outer * candidate.dt - self.target.T) > 1e-10 * max(1.0, abs(self.target.T)):
                    raise ValueError(f"e0.time_candidates[{i}].dt is not aligned with target.T")
                effective_steps.append(effective_inner_step(solver=self.target.solver, dt=candidate.dt, fine_dt=candidate.fine_dt))
            if any(not effective_steps[i] > effective_steps[i + 1] for i in range(len(effective_steps) - 1)):
                raise ValueError("e0.time_candidates effective inner steps must be strictly decreasing without duplicates")
            if any(v <= 0 for v in vars(e0.reference_tolerances).values()):
                raise ValueError("e0.reference_tolerances values must be positive")
            if any(v < 0 for v in vars(e0.algebraic_tolerances).values()):
                raise ValueError("e0.algebraic_tolerances values must be nonnegative")
            qk = (e0.q_reference_check - 1) // 2
            if e0.q_reference_check <= 0 or e0.q_reference_check % 2 == 0 or qk >= min(e0.reference_nx_candidates) / 2:
                raise ValueError("e0.q_reference_check Fourier band is not representable below Nyquist")
            identity = e0.model1_identity
            if identity.observation_dim != identity.surrogate_internal_nx:
                raise ValueError("e0.model1_identity requires observation_dim = surrogate_internal_nx")
            if identity.target_data_nx > dims.reference_nx:
                raise ValueError("e0.model1_identity.target_data_nx exceeds reference_nx")
            iqk = (identity.target_output_dim - 1) // 2
            if identity.target_output_dim <= 0 or identity.target_output_dim % 2 == 0 or iqk >= min(identity.target_data_nx, identity.observation_dim) / 2:
                raise ValueError("e0.model1_identity.target_output_dim is not representable below Nyquist")
            reduced = e0.reduced_j
            if not (1 < reduced.observation_dim < identity.surrogate_internal_nx):
                raise ValueError("e0.reduced_j.observation_dim must satisfy 1 < J < surrogate_internal_nx")
            rk = (reduced.target_output_dim - 1) // 2
            if reduced.target_output_dim <= 0 or reduced.target_output_dim % 2 == 0 or rk >= reduced.observation_dim / 2:
                raise ValueError("e0.reduced_j.target_output_dim is not representable below observation Nyquist")
            alias_k = reduced.observation_dim + max(1, rk)
            if alias_k >= identity.surrogate_internal_nx / 2:
                raise ValueError("e0.reduced_j leaves no representable high mode for aliasing counterexample")
        return self

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _strict_dataclass(cls, values: dict[str, Any], *, path: str):
    names = {f.name for f in fields(cls)}
    unknown = sorted(set(values) - names)
    if unknown:
        raise ValueError(f"unknown config key: {path}.{unknown[0]}" if path else f"unknown config key: {unknown[0]}")
    return cls(**values)


def config_from_dict(raw: dict[str, Any]) -> Paper1Config:
    top_names = {"domain", "data", "target", "spatial", "e0"}
    unknown_top = sorted(set(raw) - top_names)
    if unknown_top:
        raise ValueError(f"unknown config key: {unknown_top[0]}")
    spatial_raw = dict(raw.get("spatial", {}))
    legacy = spatial_raw.pop("target_master_nx", None)
    canonical = spatial_raw.get("reference_nx")
    if canonical is not None and legacy is not None and canonical != legacy:
        raise ValueError("spatial.reference_nx conflicts with legacy spatial.target_master_nx")
    if canonical is None and legacy is not None:
        spatial_raw["reference_nx"] = legacy
    e0_raw = raw.get("e0")
    e0 = None
    if e0_raw is not None:
        e0_values = dict(e0_raw)
        allowed = {f.name for f in fields(E0Config)}
        unknown = sorted(set(e0_values) - allowed)
        if unknown:
            raise ValueError(f"unknown config key: e0.{unknown[0]}")
        times_raw = e0_values.get("time_candidates", [])
        e0_values["time_candidates"] = tuple(
            _strict_dataclass(E0TimeCandidateConfig, dict(v), path=f"e0.time_candidates[{i}]") for i, v in enumerate(times_raw)
        )
        e0_values["calibration_sample_ids"] = tuple(e0_values.get("calibration_sample_ids", []))
        e0_values["reference_nx_candidates"] = tuple(e0_values.get("reference_nx_candidates", []))
        for key, cls in (
            ("reference_tolerances", E0ReferenceTolerancesConfig),
            ("algebraic_tolerances", E0AlgebraicTolerancesConfig),
            ("model1_identity", E0Model1IdentityConfig),
            ("reduced_j", E0ReducedJConfig),
        ):
            e0_values[key] = _strict_dataclass(cls, dict(e0_values.get(key, {})), path=f"e0.{key}")
        e0 = _strict_dataclass(E0Config, e0_values, path="e0")
    cfg = Paper1Config(
        domain=_strict_dataclass(DomainConfig, dict(raw.get("domain", {})), path="domain"),
        data=_strict_dataclass(DataConfig, dict(raw.get("data", {})), path="data"),
        target=_strict_dataclass(TargetPDEConfig, dict(raw.get("target", {})), path="target"),
        spatial=_strict_dataclass(SpatialConfig, spatial_raw, path="spatial"),
        e0=e0,
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
