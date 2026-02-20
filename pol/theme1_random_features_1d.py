from __future__ import annotations

from typing import Sequence

import numpy as np
import torch

from .elm import FixedRandomELM
from .features_1d import collect_observations, flatten_observations
from .reservoir_1d import Reservoir1DSolver, ReservoirConfig


def _sample_uniform_or_loguniform(
    rng: np.random.Generator,
    value_range: tuple[float, float],
    *,
    dist: str = "uniform",
) -> float:
    low, high = float(value_range[0]), float(value_range[1])
    if low > high:
        raise ValueError(f"invalid range: low={low} > high={high}")
    if dist == "uniform":
        return float(rng.uniform(low, high))
    if dist == "loguniform":
        if low <= 0.0 or high <= 0.0:
            raise ValueError("loguniform requires positive range bounds")
        return float(np.exp(rng.uniform(np.log(low), np.log(high))))
    raise ValueError(f"unsupported dist: {dist}")


def sample_reservoir_configs(
    *,
    reservoir: str,
    R: int,
    theta_seed: int,
    rd_nu_range: tuple[float, float] = (1e-4, 1e-2),
    rd_alpha_range: tuple[float, float] = (0.5, 1.5),
    rd_beta_range: tuple[float, float] = (0.5, 1.5),
    rd_nu_dist: str = "loguniform",
    res_burgers_nu_range: tuple[float, float] = (1e-3, 2e-1),
    res_burgers_nu_dist: str = "loguniform",
    ks_nl_range: tuple[float, float] = (0.7, 1.3),
    ks_c2_range: tuple[float, float] = (0.7, 1.3),
    ks_c4_range: tuple[float, float] = (0.7, 1.3),
    ks_dealias: bool = False,
) -> list[ReservoirConfig]:
    if R <= 0:
        raise ValueError("R must be positive")

    rng = np.random.default_rng(theta_seed)
    configs: list[ReservoirConfig] = []

    for _ in range(R):
        cfg = ReservoirConfig(reservoir=reservoir, ks_dealias=ks_dealias)
        if reservoir == "reaction_diffusion":
            cfg.rd_nu = _sample_uniform_or_loguniform(rng, rd_nu_range, dist=rd_nu_dist)
            cfg.rd_alpha = _sample_uniform_or_loguniform(rng, rd_alpha_range, dist="uniform")
            cfg.rd_beta = _sample_uniform_or_loguniform(rng, rd_beta_range, dist="uniform")
        elif reservoir == "burgers":
            cfg.res_burgers_nu = _sample_uniform_or_loguniform(
                rng,
                res_burgers_nu_range,
                dist=res_burgers_nu_dist,
            )
        elif reservoir == "ks":
            cfg.ks_nl = _sample_uniform_or_loguniform(rng, ks_nl_range, dist="uniform")
            cfg.ks_c2 = _sample_uniform_or_loguniform(rng, ks_c2_range, dist="uniform")
            cfg.ks_c4 = _sample_uniform_or_loguniform(rng, ks_c4_range, dist="uniform")
        else:
            raise ValueError(f"unsupported reservoir: {reservoir}")
        configs.append(cfg)

    return configs


class RandomReservoirFeatureMap1D:
    def __init__(
        self,
        *,
        solvers: Sequence[Reservoir1DSolver],
        obs_steps: Sequence[int],
        obs: str,
        sensor_idx: torch.Tensor,
        Tr: float,
        dt: float,
        input_scale: float = 1.0,
        input_shift: float = 0.0,
        use_elm: bool = True,
        elm_mode: str = "per_reservoir",
        elm_h_per: int = 128,
        elm_h: int = 2048,
        elm_activation: str = "tanh",
        elm_seed: int = 0,
        elm_weight_scale: float = 0.0,
        elm_bias_scale: float = 1.0,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ):
        if len(solvers) == 0:
            raise ValueError("solvers must be non-empty")
        if len(obs_steps) == 0:
            raise ValueError("obs_steps must be non-empty")
        if obs not in {"full", "points"}:
            raise ValueError("obs must be full or points")
        if elm_mode not in {"per_reservoir", "global"}:
            raise ValueError("elm_mode must be per_reservoir or global")

        self.solvers = list(solvers)
        self.obs_steps = [int(s) for s in obs_steps]
        self.obs = obs
        self.sensor_idx = sensor_idx.to(dtype=torch.long)
        self.Tr = float(Tr)
        self.dt = float(dt)
        self.input_scale = float(input_scale)
        self.input_shift = float(input_shift)
        self.use_elm = bool(use_elm)
        self.elm_mode = elm_mode
        self.elm_h_per = int(elm_h_per)
        self.elm_h = int(elm_h)
        self.device = device
        self.dtype = dtype

        self.raw_feature_dim_per_reservoir = len(self.obs_steps) * int(self.sensor_idx.numel())
        self.raw_feature_dim = len(self.solvers) * self.raw_feature_dim_per_reservoir

        self.elm_list: list[FixedRandomELM] = []
        self.elm_global: FixedRandomELM | None = None

        if self.use_elm:
            if self.elm_mode == "per_reservoir":
                if self.elm_h_per <= 0:
                    raise ValueError("elm_h_per must be positive")
                for r in range(len(self.solvers)):
                    elm = FixedRandomELM(
                        in_dim=self.raw_feature_dim_per_reservoir,
                        hidden_dim=self.elm_h_per,
                        activation=elm_activation,
                        seed=elm_seed + r,
                        weight_scale=elm_weight_scale,
                        bias_scale=elm_bias_scale,
                        device=self.device,
                        dtype=self.dtype,
                    )
                    self.elm_list.append(elm)
                self.final_feature_dim = len(self.solvers) * self.elm_h_per
            else:
                if self.elm_h <= 0:
                    raise ValueError("elm_h must be positive")
                self.elm_global = FixedRandomELM(
                    in_dim=self.raw_feature_dim,
                    hidden_dim=self.elm_h,
                    activation=elm_activation,
                    seed=elm_seed,
                    weight_scale=elm_weight_scale,
                    bias_scale=elm_bias_scale,
                    device=self.device,
                    dtype=self.dtype,
                )
                self.final_feature_dim = self.elm_h
        else:
            self.final_feature_dim = self.raw_feature_dim

    @torch.no_grad()
    def __call__(self, x_batch: torch.Tensor) -> torch.Tensor:
        x = x_batch.to(device=self.device, dtype=self.dtype)
        z0 = self.input_scale * x + self.input_shift
        sensor_idx = self.sensor_idx.to(x.device)

        reservoir_features: list[torch.Tensor] = []
        transformed_features: list[torch.Tensor] = []
        for r, solver in enumerate(self.solvers):
            states = solver.simulate(z0, dt=self.dt, Tr=self.Tr, obs_steps=self.obs_steps)
            obs_list = collect_observations(states, self.obs, sensor_idx)
            phi_r = flatten_observations(obs_list)
            reservoir_features.append(phi_r)

            if self.use_elm and self.elm_mode == "per_reservoir":
                transformed_features.append(self.elm_list[r](phi_r))

        if self.use_elm:
            if self.elm_mode == "per_reservoir":
                return torch.cat(transformed_features, dim=-1)
            phi = torch.cat(reservoir_features, dim=-1)
            return self.elm_global(phi)

        return torch.cat(reservoir_features, dim=-1)
