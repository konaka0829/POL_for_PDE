from __future__ import annotations

from dataclasses import dataclass, replace
import time
from typing import Iterable, Sequence

import torch

from pol.elm import FixedRandomELM
from pol.features_1d import (
    build_sensor_indices,
    build_time_grid,
    collect_observations,
    flatten_observations,
)
from pol.reservoir_1d import Reservoir1DSolver, ReservoirConfig
from pol.ridge import fit_ridge_streaming, fit_ridge_streaming_standardized, predict_linear


def _resolve_device(device: str | torch.device) -> torch.device:
    if isinstance(device, torch.device):
        return device
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        return torch.device("cuda")
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cpu")


@dataclass(frozen=True)
class Model123Config:
    reservoir: str = "burgers"
    Ttilde: float = 1.0
    dt: float = 1e-2
    K: int = 1
    feature_times: str = ""
    obs: str = "full"
    J: int = 128
    sensor_mode: str = "equispaced"
    sensor_seed: int = 0
    input_scale: float = 1.0
    input_shift: float = 0.0
    ridge_lambda: float = 1e-4
    ridge_dtype: torch.dtype = torch.float64
    standardize_features: bool = False
    feature_std_eps: float = 1e-6
    elm_hidden_dim: int = 1024
    elm_activation: str = "tanh"
    elm_seed: int = 0
    elm_weight_scale: float = 0.0
    elm_bias_scale: float = 1.0
    rd_nu: float = 1e-3
    rd_alpha: float = 1.0
    rd_beta: float = 1.0
    res_burgers_nu: float = 5e-2
    res_burgers_b: float = 1.0
    ks_dealias: bool = False
    ks_b: float = 1.0
    ks_eta: float = 1.0
    ks_kappa: float = 1.0
    burgers_scheme: str = "split_step"
    burgers_fine_dt: float = 1e-4
    burgers_dealias: bool = True
    device: str | torch.device = "cpu"
    dtype: torch.dtype = torch.float32


@dataclass(frozen=True)
class ObservationSlices:
    starts: list[int]
    stops: list[int]

    def last(self) -> slice:
        return slice(self.starts[-1], self.stops[-1])

    def at(self, index: int) -> slice:
        return slice(self.starts[index], self.stops[index])


class ObservedTrajectoryFeature1D:
    def __init__(self, *, s: int, config: Model123Config):
        if s <= 1:
            raise ValueError("s must be >= 2")
        self.s = int(s)
        self.config = config
        self.device = _resolve_device(config.device)
        self.dtype = config.dtype
        self.times, self.obs_steps = build_time_grid(
            Tr=config.Ttilde,
            dt=config.dt,
            K=config.K,
            feature_times=config.feature_times,
        )
        self.operator = build_sensor_indices(
            s=self.s,
            obs=config.obs,
            J=config.J,
            sensor_mode=config.sensor_mode,
            sensor_seed=config.sensor_seed,
        )
        self.reservoir = Reservoir1DSolver(
            ReservoirConfig(
                reservoir=config.reservoir,
                rd_nu=config.rd_nu,
                rd_alpha=config.rd_alpha,
                rd_beta=config.rd_beta,
                res_burgers_nu=config.res_burgers_nu,
                res_burgers_b=config.res_burgers_b,
                ks_dealias=config.ks_dealias,
                ks_b=config.ks_b,
                ks_eta=config.ks_eta,
                ks_kappa=config.ks_kappa,
                burgers_scheme=config.burgers_scheme,
                burgers_fine_dt=config.burgers_fine_dt,
                burgers_dealias=config.burgers_dealias,
            )
        )

    def encode(self, u0_batch: torch.Tensor) -> torch.Tensor:
        x = u0_batch.to(device=self.device, dtype=self.dtype)
        if x.ndim != 2 or x.shape[-1] != self.s:
            raise ValueError(f"u0_batch must have shape (B, {self.s}), got {tuple(x.shape)}")
        return self.config.input_scale * x + self.config.input_shift

    @torch.no_grad()
    def simulate_states(self, u0_batch: torch.Tensor) -> list[torch.Tensor]:
        z0 = self.encode(u0_batch)
        return self.reservoir.simulate(
            z0,
            dt=self.config.dt,
            Tr=self.config.Ttilde,
            obs_steps=self.obs_steps,
        )

    @torch.no_grad()
    def simulate_state_at_time(self, u0_batch: torch.Tensor, t: float) -> torch.Tensor:
        if t <= 0.0:
            raise ValueError("t must be positive")
        step = max(1, int(round(float(t) / self.config.dt)))
        z0 = self.encode(u0_batch)
        return self.reservoir.simulate(
            z0,
            dt=self.config.dt,
            Tr=float(t),
            obs_steps=[step],
        )[0]

    @torch.no_grad()
    def collect_observations(self, states: Sequence[torch.Tensor]) -> list[torch.Tensor]:
        return collect_observations(states, self.config.obs, self.operator.to(self.device))

    @torch.no_grad()
    def flatten(self, observations: Sequence[torch.Tensor]) -> torch.Tensor:
        return flatten_observations(observations)

    @torch.no_grad()
    def __call__(self, u0_batch: torch.Tensor) -> torch.Tensor:
        return self.flatten(self.collect_observations(self.simulate_states(u0_batch)))

    def observation_width(self) -> int:
        if self.config.obs == "full":
            return self.s
        if self.config.obs == "fourier":
            return 2 * int(self.operator.shape[0])
        if self.config.obs == "proj":
            return int(self.operator.shape[0])
        return int(self.operator.shape[0])

    def observation_slices(self) -> ObservationSlices:
        width = self.observation_width()
        starts = [k * width for k in range(len(self.obs_steps))]
        stops = [(k + 1) * width for k in range(len(self.obs_steps))]
        return ObservationSlices(starts=starts, stops=stops)


class FiniteDimObservation1D:
    def __init__(self, trajectory: ObservedTrajectoryFeature1D):
        self.trajectory = trajectory

    @torch.no_grad()
    def __call__(self, u0_batch: torch.Tensor) -> torch.Tensor:
        return self.trajectory(u0_batch)

    def slices(self) -> ObservationSlices:
        return self.trajectory.observation_slices()

    def select_last_observation(self, phi: torch.Tensor) -> torch.Tensor:
        return phi[:, self.slices().last()]


class Model1Predictor1D:
    def __init__(self, *, s: int, config: Model123Config):
        self.s = int(s)
        self.config = replace(config, obs="full", J=s)
        self.feature_map = ObservedTrajectoryFeature1D(s=s, config=self.config)

    @torch.no_grad()
    def predict(self, u0_batch: torch.Tensor) -> torch.Tensor:
        return self.feature_map.simulate_state_at_time(u0_batch, self.config.Ttilde)


class Model2Regressor1D:
    def __init__(self, *, s: int, config: Model123Config):
        self.s = int(s)
        self.config = config
        self.feature_map = ObservedTrajectoryFeature1D(s=s, config=config)
        self.observation = FiniteDimObservation1D(self.feature_map)
        self.weight: torch.Tensor | None = None
        self.ridge_state: dict[str, torch.Tensor] | None = None

    @torch.no_grad()
    def features(self, u0_batch: torch.Tensor) -> torch.Tensor:
        return self.observation(u0_batch)

    def fit(
        self,
        train_loader: Iterable,
        *,
        progress_fn=None,
        progress_label: str | None = None,
    ) -> dict[str, torch.Tensor]:
        if progress_label:
            print("[%s] ridge feature accumulation start" % progress_label, flush=True)
        start_time = time.perf_counter()
        if self.config.standardize_features:
            ridge_state = fit_ridge_streaming_standardized(
                train_loader,
                self.features,
                self.config.ridge_lambda,
                dtype=self.config.ridge_dtype,
                regularize_bias=False,
                eps=self.config.feature_std_eps,
                progress_fn=progress_fn,
            )
        else:
            ridge_state = fit_ridge_streaming(
                train_loader,
                self.features,
                self.config.ridge_lambda,
                dtype=self.config.ridge_dtype,
                regularize_bias=False,
                progress_fn=progress_fn,
            )
        if progress_label:
            elapsed = time.perf_counter() - start_time
            print("[%s] ridge solve done in %.2fs" % (progress_label, elapsed), flush=True)
        self.weight = ridge_state["W"]
        self.ridge_state = ridge_state
        return ridge_state

    @torch.no_grad()
    def predict(self, u0_batch: torch.Tensor) -> torch.Tensor:
        if self.weight is None:
            raise RuntimeError("Model2Regressor1D.fit must be called before predict")
        feat = self.features(u0_batch).to(dtype=self.weight.dtype)
        return predict_linear(feat, self.weight)


class Model3Regressor1D:
    def __init__(self, *, s: int, config: Model123Config):
        self.s = int(s)
        self.config = config
        self.feature_map = ObservedTrajectoryFeature1D(s=s, config=config)
        self.observation = FiniteDimObservation1D(self.feature_map)
        self.elm: FixedRandomELM | None = None
        self.weight: torch.Tensor | None = None
        self.ridge_state: dict[str, torch.Tensor] | None = None

    @torch.no_grad()
    def phi(self, u0_batch: torch.Tensor) -> torch.Tensor:
        return self.observation(u0_batch)

    def _ensure_elm(self, in_dim: int) -> FixedRandomELM:
        if self.elm is None:
            self.elm = FixedRandomELM(
                in_dim=in_dim,
                hidden_dim=self.config.elm_hidden_dim,
                activation=self.config.elm_activation,
                seed=self.config.elm_seed,
                weight_scale=self.config.elm_weight_scale,
                bias_scale=self.config.elm_bias_scale,
                device=self.feature_map.device,
                dtype=self.feature_map.dtype,
            )
        return self.elm

    @torch.no_grad()
    def augment_features(self, phi: torch.Tensor) -> torch.Tensor:
        elm = self._ensure_elm(phi.shape[1])
        h = elm(phi)
        return torch.cat([phi, h], dim=-1)

    @torch.no_grad()
    def features(self, u0_batch: torch.Tensor) -> torch.Tensor:
        phi = self.phi(u0_batch)
        return self.augment_features(phi)

    def fit(
        self,
        train_loader: Iterable,
        *,
        progress_fn=None,
        progress_label: str | None = None,
    ) -> dict[str, torch.Tensor]:
        fit_start = time.perf_counter()
        if progress_label:
            print("[%s] probe batch for ELM init" % progress_label, flush=True)
        probe_batch = next(iter(train_loader))[0]
        probe_phi = self.phi(probe_batch)
        self._ensure_elm(probe_phi.shape[1])
        if progress_label:
            print(
                "[%s] ELM ready (in=%d, hidden=%d)"
                % (progress_label, probe_phi.shape[1], self.config.elm_hidden_dim),
                flush=True,
            )
        if self.config.standardize_features:
            ridge_state = fit_ridge_streaming_standardized(
                train_loader,
                self.features,
                self.config.ridge_lambda,
                dtype=self.config.ridge_dtype,
                regularize_bias=False,
                eps=self.config.feature_std_eps,
                progress_fn=progress_fn,
            )
        else:
            ridge_state = fit_ridge_streaming(
                train_loader,
                self.features,
                self.config.ridge_lambda,
                dtype=self.config.ridge_dtype,
                regularize_bias=False,
                progress_fn=progress_fn,
            )
        if progress_label:
            elapsed = time.perf_counter() - fit_start
            print("[%s] ridge solve done in %.2fs" % (progress_label, elapsed), flush=True)
        self.weight = ridge_state["W"]
        self.ridge_state = ridge_state
        return ridge_state

    @torch.no_grad()
    def predict(self, u0_batch: torch.Tensor) -> torch.Tensor:
        if self.weight is None:
            raise RuntimeError("Model3Regressor1D.fit must be called before predict")
        feat = self.features(u0_batch).to(dtype=self.weight.dtype)
        return predict_linear(feat, self.weight)
