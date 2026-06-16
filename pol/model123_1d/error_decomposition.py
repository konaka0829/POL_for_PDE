from __future__ import annotations

import csv
import json
import math
import os
from dataclasses import asdict, dataclass
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from pol.burgers_spectral_1d import make_wavenumbers, simulate_burgers_split_step
from pol.model123_1d.initial_conditions import (
    evaluate_initial_conditions,
    sample_gaussian_random_field_initial_conditions,
    sample_initial_condition_coefficients,
)
from pol.model123_1d.metrics import discrete_l2h_norm
from pol.plotting import save_figure_all_formats
from pol.reservoir_1d import Reservoir1DSolver, ReservoirConfig


@dataclass(frozen=True)
class ErrorDecompositionConfig:
    num_samples: int = 64
    nx: int = 256
    seed: int = 0
    batch_size: int = 8
    target_nu: float = 0.05
    domain_length: float = 1.0
    T: float = 1.0
    Ttilde_values: list[float] | tuple[float, ...] = (1.0,)
    dt: float = 1e-2
    fine_dt: float = 1e-3
    reservoir: str = "burgers"
    rd_nu: float = 1e-3
    rd_alpha: float = 1.0
    rd_beta: float = 1.0
    res_burgers_nu: float = 0.05
    res_burgers_b: float = 1.0
    burgers_scheme: str = "split_step"
    burgers_dealias: bool = False
    ks_b: float = 1.0
    ks_eta: float = 1.0
    ks_kappa: float = 1.0
    ks_dealias: bool = False
    input_scale: float = 1.0
    input_shift: float = 0.0
    dtype: str = "float64"
    device: str = "cpu"
    initial_condition_type: str = "fourier"
    grf_gamma: float = 2.0
    grf_tau: float = 5.0
    grf_sigma: float = 25.0
    grf_mean: float = 0.0
    beta_mode: str = "zero"
    beta_fixed: float = 0.0
    beta_pairwise_margin: float = 0.0
    beta_max_states: int = 24
    calibration_num_samples: int = 0
    calibration_seed: int | None = None
    time_quadrature: str = "trapezoid"
    out_dir: str = "outputs/model1_error_decomposition_1d"


def _resolve_device(name: str) -> torch.device:
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        return torch.device("cuda")
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cpu")


def _resolve_dtype(name: str) -> torch.dtype:
    if name == "float32":
        return torch.float32
    return torch.float64


def _config_to_jsonable(cfg: ErrorDecompositionConfig) -> dict[str, Any]:
    return asdict(cfg)


def _validate_config(cfg: ErrorDecompositionConfig) -> None:
    if cfg.num_samples <= 0:
        raise ValueError("num_samples must be positive")
    if cfg.nx <= 1:
        raise ValueError("nx must be >= 2")
    if cfg.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if cfg.target_nu < 0.0:
        raise ValueError("target_nu must be non-negative")
    if cfg.domain_length <= 0.0:
        raise ValueError("domain_length must be positive")
    if cfg.T <= 0.0:
        raise ValueError("T must be positive")
    if cfg.dt <= 0.0:
        raise ValueError("dt must be positive")
    if cfg.fine_dt <= 0.0:
        raise ValueError("fine_dt must be positive")
    if cfg.reservoir not in {"burgers", "reaction_diffusion", "ks"}:
        raise ValueError(f"Unsupported reservoir family: {cfg.reservoir}")
    if cfg.burgers_scheme not in {"semi_implicit", "split_step"}:
        raise ValueError("burgers_scheme must be semi_implicit or split_step")
    if cfg.initial_condition_type not in {"fourier", "grf"}:
        raise ValueError("initial_condition_type must be 'fourier' or 'grf'")
    if cfg.beta_mode not in {"zero", "analytic_safe", "analytic_safe_poincare", "empirical_pairwise", "fixed"}:
        raise ValueError("unsupported beta_mode")
    if cfg.beta_max_states <= 1:
        raise ValueError("beta_max_states must be >= 2")
    if cfg.calibration_num_samples < 0:
        raise ValueError("calibration_num_samples must be >= 0")
    if cfg.time_quadrature not in {"trapezoid", "left"}:
        raise ValueError("time_quadrature must be trapezoid or left")
    if not cfg.Ttilde_values:
        raise ValueError("Ttilde_values must be non-empty")
    for value in cfg.Ttilde_values:
        if value <= 0.0:
            raise ValueError("All Ttilde values must be positive")


def make_initial_conditions(
    cfg: ErrorDecompositionConfig,
    *,
    num_samples: int | None = None,
    seed: int | None = None,
) -> torch.Tensor:
    device = _resolve_device(cfg.device)
    dtype = _resolve_dtype(cfg.dtype)
    sample_count = cfg.num_samples if num_samples is None else int(num_samples)
    sample_seed = cfg.seed if seed is None else int(seed)
    if cfg.initial_condition_type == "fourier":
        coeffs = sample_initial_condition_coefficients(sample_count, seed=sample_seed, dtype=dtype)
        return evaluate_initial_conditions(coeffs, cfg.nx, device=device, dtype=dtype).cpu()
    return sample_gaussian_random_field_initial_conditions(
        sample_count,
        cfg.nx,
        seed=sample_seed,
        gamma=cfg.grf_gamma,
        tau=cfg.grf_tau,
        sigma=cfg.grf_sigma,
        mean=cfg.grf_mean,
        device=device,
        dtype=dtype,
    ).cpu()


def _num_steps_for_time(time_value: float, dt: float) -> int:
    return int(math.ceil(float(time_value) / float(dt) - 1e-12))


def _simulate_target_trajectory(u0: torch.Tensor, cfg: ErrorDecompositionConfig) -> torch.Tensor:
    total_steps = int(round(cfg.T / cfg.dt))
    if not math.isclose(total_steps * cfg.dt, cfg.T, rel_tol=1e-9, abs_tol=1e-12):
        raise ValueError("cfg.T must align with cfg.dt for the target grid")
    obs_steps = list(range(1, total_steps + 1))
    states_per_step: list[list[torch.Tensor]] = [[] for _ in obs_steps]
    work_device = _resolve_device(cfg.device)
    dtype = _resolve_dtype(cfg.dtype)
    for start in range(0, u0.shape[0], cfg.batch_size):
        batch = u0[start : start + cfg.batch_size].to(device=work_device, dtype=dtype)
        states = simulate_burgers_split_step(
            batch,
            dt=cfg.dt,
            Tr=cfg.T,
            obs_steps=obs_steps,
            nu=cfg.target_nu,
            fine_dt=cfg.fine_dt,
            b=1.0,
            forcing=None,
            forcing_steps=None,
            dealias=False,
            domain_length=cfg.domain_length,
        )
        for idx, state in enumerate(states):
            states_per_step[idx].append(state.detach().cpu())
    stacked = torch.stack([torch.cat(chunks, dim=0) for chunks in states_per_step], dim=0)
    return torch.cat([u0.unsqueeze(0).cpu(), stacked], dim=0)


def _make_surrogate_solver(cfg: ErrorDecompositionConfig) -> Reservoir1DSolver:
    if cfg.reservoir == "burgers":
        return Reservoir1DSolver(
            ReservoirConfig(
                reservoir="burgers",
                res_burgers_nu=cfg.res_burgers_nu,
                res_burgers_b=cfg.res_burgers_b,
                burgers_scheme=cfg.burgers_scheme,
                burgers_fine_dt=cfg.fine_dt,
                burgers_dealias=cfg.burgers_dealias,
                domain_length=cfg.domain_length,
            )
        )
    if cfg.reservoir == "reaction_diffusion":
        return Reservoir1DSolver(
            ReservoirConfig(
                reservoir="reaction_diffusion",
                rd_nu=cfg.rd_nu,
                rd_alpha=cfg.rd_alpha,
                rd_beta=cfg.rd_beta,
                domain_length=cfg.domain_length,
            )
        )
    return Reservoir1DSolver(
        ReservoirConfig(
            reservoir="ks",
            ks_b=cfg.ks_b,
            ks_eta=cfg.ks_eta,
            ks_kappa=cfg.ks_kappa,
            ks_dealias=cfg.ks_dealias,
            domain_length=cfg.domain_length,
        )
    )


def _simulate_surrogate_trajectory(u0: torch.Tensor, cfg: ErrorDecompositionConfig) -> torch.Tensor:
    max_time = max(float(value) for value in cfg.Ttilde_values)
    total_steps = _num_steps_for_time(max_time, cfg.dt)
    obs_steps = list(range(1, total_steps + 1))
    states_per_step: list[list[torch.Tensor]] = [[] for _ in obs_steps]
    work_device = _resolve_device(cfg.device)
    dtype = _resolve_dtype(cfg.dtype)
    solver = _make_surrogate_solver(cfg)
    integration_time = total_steps * cfg.dt
    for start in range(0, u0.shape[0], cfg.batch_size):
        batch = u0[start : start + cfg.batch_size].to(device=work_device, dtype=dtype)
        states = solver.simulate(batch, dt=cfg.dt, Tr=integration_time, obs_steps=obs_steps)
        for idx, state in enumerate(states):
            states_per_step[idx].append(state.detach().cpu())
    stacked = torch.stack([torch.cat(chunks, dim=0) for chunks in states_per_step], dim=0)
    return torch.cat([u0.unsqueeze(0).cpu(), stacked], dim=0)


def discrete_l2_h(values: torch.Tensor, *, domain_length: float = 1.0) -> torch.Tensor:
    return discrete_l2h_norm(values, domain_length=domain_length)


def discrete_inner_h(a: torch.Tensor, b: torch.Tensor, *, domain_length: float = 1.0) -> torch.Tensor:
    h = float(domain_length) / float(a.shape[-1])
    return h * torch.sum(a * b, dim=-1)


def spectral_derivatives_1d(z: torch.Tensor, *, domain_length: float = 1.0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    s = z.shape[-1]
    k = make_wavenumbers(s, z.device, z.dtype, domain_length=domain_length)
    z_hat = torch.fft.rfft(z, dim=-1)
    ux = torch.fft.irfft((1j * k) * z_hat, n=s, dim=-1)
    uxx = torch.fft.irfft(-(k.pow(2)) * z_hat, n=s, dim=-1)
    uxxxx = torch.fft.irfft((k.pow(4)) * z_hat, n=s, dim=-1)
    return ux, uxx, uxxxx


def burgers_generator(z: torch.Tensor, *, nu: float, domain_length: float = 1.0) -> torch.Tensor:
    ux, uxx, _ = spectral_derivatives_1d(z, domain_length=domain_length)
    return nu * uxx - z * ux


def scaled_defect_burgers_reservoir(
    z: torch.Tensor,
    *,
    alpha: float,
    target_nu: float,
    res_burgers_nu: float,
    res_burgers_b: float,
    domain_length: float = 1.0,
) -> torch.Tensor:
    ux, uxx, _ = spectral_derivatives_1d(z, domain_length=domain_length)
    return (target_nu - alpha * res_burgers_nu) * uxx + (alpha * res_burgers_b - 1.0) * z * ux


def scaled_defect_reaction_diffusion_reservoir(
    z: torch.Tensor,
    *,
    alpha: float,
    target_nu: float,
    rd_nu: float,
    rd_alpha: float,
    rd_beta: float,
    domain_length: float = 1.0,
) -> torch.Tensor:
    ux, uxx, _ = spectral_derivatives_1d(z, domain_length=domain_length)
    return (target_nu - alpha * rd_nu) * uxx - z * ux - alpha * rd_alpha * z + alpha * rd_beta * z.pow(3)


def scaled_defect_ks_reservoir(
    z: torch.Tensor,
    *,
    alpha: float,
    target_nu: float,
    ks_b: float,
    ks_eta: float,
    ks_kappa: float,
    domain_length: float = 1.0,
) -> torch.Tensor:
    ux, uxx, uxxxx = spectral_derivatives_1d(z, domain_length=domain_length)
    return (target_nu + alpha * ks_eta) * uxx + (alpha * ks_b - 1.0) * z * ux + alpha * ks_kappa * uxxxx


def scaled_generator_defect(z: torch.Tensor, cfg: ErrorDecompositionConfig, *, alpha: float) -> torch.Tensor:
    if cfg.reservoir == "burgers":
        return scaled_defect_burgers_reservoir(
            z,
            alpha=alpha,
            target_nu=cfg.target_nu,
            res_burgers_nu=cfg.res_burgers_nu,
            res_burgers_b=cfg.res_burgers_b,
            domain_length=cfg.domain_length,
        )
    if cfg.reservoir == "reaction_diffusion":
        return scaled_defect_reaction_diffusion_reservoir(
            z,
            alpha=alpha,
            target_nu=cfg.target_nu,
            rd_nu=cfg.rd_nu,
            rd_alpha=cfg.rd_alpha,
            rd_beta=cfg.rd_beta,
            domain_length=cfg.domain_length,
        )
    return scaled_defect_ks_reservoir(
        z,
        alpha=alpha,
        target_nu=cfg.target_nu,
        ks_b=cfg.ks_b,
        ks_eta=cfg.ks_eta,
        ks_kappa=cfg.ks_kappa,
        domain_length=cfg.domain_length,
    )


def defect_burgers_reservoir(
    z: torch.Tensor,
    *,
    target_nu: float,
    res_burgers_nu: float,
    res_burgers_b: float,
    domain_length: float = 1.0,
) -> torch.Tensor:
    return scaled_defect_burgers_reservoir(
        z,
        alpha=1.0,
        target_nu=target_nu,
        res_burgers_nu=res_burgers_nu,
        res_burgers_b=res_burgers_b,
        domain_length=domain_length,
    )


def defect_reaction_diffusion_reservoir(
    z: torch.Tensor,
    *,
    target_nu: float,
    rd_nu: float,
    rd_alpha: float,
    rd_beta: float,
    domain_length: float = 1.0,
) -> torch.Tensor:
    return scaled_defect_reaction_diffusion_reservoir(
        z,
        alpha=1.0,
        target_nu=target_nu,
        rd_nu=rd_nu,
        rd_alpha=rd_alpha,
        rd_beta=rd_beta,
        domain_length=domain_length,
    )


def defect_ks_reservoir(
    z: torch.Tensor,
    *,
    target_nu: float,
    ks_b: float,
    ks_eta: float,
    ks_kappa: float,
    domain_length: float = 1.0,
) -> torch.Tensor:
    return scaled_defect_ks_reservoir(
        z,
        alpha=1.0,
        target_nu=target_nu,
        ks_b=ks_b,
        ks_eta=ks_eta,
        ks_kappa=ks_kappa,
        domain_length=domain_length,
    )


def generator_defect(z: torch.Tensor, cfg: ErrorDecompositionConfig) -> torch.Tensor:
    """Deprecated compatibility alias for the unscaled alpha=1 residual."""
    return scaled_generator_defect(z, cfg, alpha=1.0)


def make_time_quadrature_weights(num_steps: int, dt: float, rule: str = "trapezoid") -> torch.Tensor:
    if num_steps < 0:
        raise ValueError("num_steps must be >= 0")
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    if rule == "trapezoid":
        weights = torch.full((num_steps + 1,), float(dt), dtype=torch.float64)
        if num_steps >= 1:
            weights[0] = 0.5 * float(dt)
            weights[-1] = 0.5 * float(dt)
        return weights
    if rule == "left":
        weights = torch.zeros((num_steps + 1,), dtype=torch.float64)
        if num_steps >= 1:
            weights[:-1] = float(dt)
        return weights
    raise ValueError("rule must be trapezoid or left")


def interpolate_trajectory_at_times(
    states: torch.Tensor,
    query_times: torch.Tensor,
    *,
    dt: float,
) -> torch.Tensor:
    """Linearly interpolate a trajectory with states[n] at native time n*dt."""
    if states.ndim < 2:
        raise ValueError("states must have a leading time dimension")
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    query = query_times.to(dtype=torch.float64, device=states.device)
    native = query / float(dt)
    lower = torch.floor(native + 1e-12).to(dtype=torch.long)
    upper = torch.clamp(lower + 1, max=states.shape[0] - 1)
    lower = torch.clamp(lower, min=0, max=states.shape[0] - 1)
    if torch.any(native < -1e-10) or torch.any(native > states.shape[0] - 1 + 1e-10):
        raise ValueError("query times exceed simulated surrogate trajectory")
    frac = (native - lower.to(dtype=torch.float64)).clamp(0.0, 1.0).to(dtype=states.dtype)
    view_shape = (frac.shape[0],) + (1,) * (states.ndim - 1)
    frac_view = frac.reshape(view_shape)
    return (1.0 - frac_view) * states.index_select(0, lower) + frac_view * states.index_select(0, upper)


def target_time_grid(cfg: ErrorDecompositionConfig) -> torch.Tensor:
    step_T = int(round(cfg.T / cfg.dt))
    if not math.isclose(step_T * cfg.dt, cfg.T, rel_tol=1e-9, abs_tol=1e-12):
        raise ValueError("cfg.T must align with cfg.dt for the target grid")
    return torch.arange(step_T + 1, dtype=torch.float64) * float(cfg.dt)


def rescaled_surrogate_states(
    surrogate_states: torch.Tensor,
    cfg: ErrorDecompositionConfig,
    *,
    alpha: float,
) -> torch.Tensor:
    s_grid = target_time_grid(cfg)
    return interpolate_trajectory_at_times(surrogate_states, alpha * s_grid, dt=cfg.dt)


def _flatten_state_pool(states: torch.Tensor, *, max_states: int) -> torch.Tensor:
    flat = states.reshape(-1, states.shape[-1])
    if flat.shape[0] <= max_states:
        return flat
    keep = torch.linspace(0, flat.shape[0] - 1, steps=max_states, dtype=torch.float64).round().long()
    return flat.index_select(0, keep)


def compute_beta(
    *,
    calibration_target_states: torch.Tensor,
    calibration_surrogate_states: torch.Tensor,
    cfg: ErrorDecompositionConfig,
) -> tuple[float, dict[str, Any]]:
    if calibration_target_states.shape != calibration_surrogate_states.shape:
        raise ValueError("beta calibration target and surrogate states must have the same shape")
    target_shared = calibration_target_states
    surrogate_shared = calibration_surrogate_states
    details: dict[str, Any] = {"mode": cfg.beta_mode}

    if cfg.beta_mode == "zero":
        details["chosen_beta"] = 0.0
        return 0.0, details

    if cfg.beta_mode == "fixed":
        details["chosen_beta"] = float(cfg.beta_fixed)
        return float(cfg.beta_fixed), details

    if cfg.beta_mode in {"analytic_safe", "analytic_safe_poincare"}:
        combined = torch.cat([target_shared, surrogate_shared], dim=1)
        ux, _, _ = spectral_derivatives_1d(combined.reshape(-1, combined.shape[-1]), domain_length=cfg.domain_length)
        M_K_hat = float(ux.abs().amax().item())
        beta = 0.5 * M_K_hat
        details["M_K_hat"] = M_K_hat
        if cfg.beta_mode == "analytic_safe_poincare":
            target_means = torch.mean(target_shared, dim=-1)
            surrogate_means = torch.mean(surrogate_shared, dim=-1)
            if not torch.allclose(target_means, surrogate_means, atol=1e-8, rtol=1e-6):
                raise ValueError("analytic_safe_poincare requires samplewise means to match")
            beta = beta - cfg.target_nu * (2.0 * math.pi) ** 2
            details["poincare_shift"] = -cfg.target_nu * (2.0 * math.pi / cfg.domain_length) ** 2
        details["chosen_beta"] = float(beta)
        return float(beta), details

    flat = _flatten_state_pool(torch.cat([target_shared, surrogate_shared], dim=1), max_states=cfg.beta_max_states)
    F = burgers_generator(flat, nu=cfg.target_nu, domain_length=cfg.domain_length)
    pairwise_max = -float("inf")
    pairs_used = 0
    for i in range(flat.shape[0]):
        zi = flat[i : i + 1]
        Fi = F[i : i + 1]
        for j in range(i):
            dz = zi - flat[j : j + 1]
            denom = float(discrete_inner_h(dz, dz, domain_length=cfg.domain_length).item())
            if denom <= 1e-14:
                continue
            dF = Fi - F[j : j + 1]
            ratio = float(discrete_inner_h(dF, dz, domain_length=cfg.domain_length).item() / denom)
            pairwise_max = max(pairwise_max, ratio)
            pairs_used += 1
    if pairwise_max == -float("inf"):
        pairwise_max = 0.0
    beta = pairwise_max + float(cfg.beta_pairwise_margin)
    details["pairwise_max"] = float(pairwise_max)
    details["pairwise_margin"] = float(cfg.beta_pairwise_margin)
    details["pairs_used"] = int(pairs_used)
    details["chosen_beta"] = float(beta)
    return float(beta), details


def estimate_empirical_beta(
    target_states: torch.Tensor,
    surrogate_states: torch.Tensor,
    cfg: ErrorDecompositionConfig,
) -> float:
    beta_mode = cfg.beta_mode
    if beta_mode != "empirical_pairwise":
        cfg = ErrorDecompositionConfig(**{**asdict(cfg), "beta_mode": "empirical_pairwise"})
    beta, _ = compute_beta(
        calibration_target_states=target_states,
        calibration_surrogate_states=surrogate_states,
        cfg=cfg,
    )
    return beta


def c_beta_T(beta: float, T: float) -> float:
    if abs(beta) < 1e-12:
        return math.sqrt(T)
    value = (math.exp(2.0 * beta * T) - 1.0) / (2.0 * beta)
    return math.sqrt(max(value, 0.0))


def _rms(values: list[float]) -> float:
    arr = np.asarray(values, dtype=float)
    return float(np.sqrt(np.mean(arr * arr)))


def aggregate_metric_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[float, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(float(row["Ttilde"]), []).append(row)

    summary_rows: list[dict[str, Any]] = []
    for ttilde in sorted(grouped):
        t_rows = grouped[ttilde]
        delta_init = _rms([float(row["Delta_init_abs_l2h"]) for row in t_rows])
        delta_scale = _rms([float(row["Delta_scale_abs_l2h"]) for row in t_rows])
        d1 = _rms([float(row["D1_abs_l2h"]) for row in t_rows])
        beta_value = float(t_rows[0]["beta_value"])
        cbeta = float(t_rows[0]["c_beta_T"])
        T = float(t_rows[0]["T"])
        exp_beta_t = math.exp(beta_value * T)
        rhs_beta0 = delta_init + math.sqrt(T) * delta_scale
        rhs_beta_theorem_components = exp_beta_t * delta_init + cbeta * delta_scale
        pathwise_rms = _rms([float(row["rhs_beta_pathwise_abs_l2h"]) for row in t_rows])
        summary_rows.append(
            {
                "Ttilde": float(ttilde),
                "alpha": float(t_rows[0]["alpha"]),
                "num_samples": len(t_rows),
                "D1": d1,
                "Delta_init": delta_init,
                "Delta_scale": delta_scale,
                "Delta_dyn": delta_scale,
                "rhs_beta0": rhs_beta0,
                "rhs_beta": rhs_beta_theorem_components,
                "rhs_beta_legacy_alias_of": "rhs_beta_theorem_components",
                "rhs_beta_theorem_components": rhs_beta_theorem_components,
                "rhs_beta_pathwise_rms": pathwise_rms,
                "beta_mode": t_rows[0]["beta_mode"],
                "beta_value": beta_value,
                "beta_empirical": beta_value,
                "c_beta_T": cbeta,
            }
        )
    return summary_rows


def _dataset_beta_value(cfg: ErrorDecompositionConfig, beta_value: float | None) -> float:
    if beta_value is not None:
        return float(beta_value)
    if cfg.beta_mode == "zero":
        return 0.0
    if cfg.beta_mode == "fixed":
        return float(cfg.beta_fixed)
    raise ValueError(
        "compute_time_scaled_defect_for_dataset currently supports beta_mode='zero', "
        "beta_mode='fixed', or an explicit beta_value. Other beta modes require full "
        "target trajectories for calibration."
    )


def _summary_for_dataset_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    delta_values = np.asarray([float(row["delta_scale_pathwise_abs_l2h"]) for row in rows], dtype=float)
    d1_values = np.asarray([float(row["D1_model1_abs_l2h"]) for row in rows], dtype=float)
    delta_init_values = np.asarray([float(row["Delta_init_abs_l2h"]) for row in rows], dtype=float)
    rhs_beta_values = np.asarray([float(row["rhs_beta_pathwise_abs_l2h"]) for row in rows], dtype=float)
    rhs_beta0_values = np.asarray([float(row["rhs_beta0_pathwise_abs_l2h"]) for row in rows], dtype=float)
    first = rows[0]
    delta_rms = float(np.sqrt(np.mean(delta_values * delta_values)))
    d1_rms = float(np.sqrt(np.mean(d1_values * d1_values)))
    delta_init_rms = float(np.sqrt(np.mean(delta_init_values * delta_init_values)))
    rhs_beta_rms = float(np.sqrt(np.mean(rhs_beta_values * rhs_beta_values)))
    rhs_beta0_rms = float(np.sqrt(np.mean(rhs_beta0_values * rhs_beta0_values)))
    return {
        "T": float(first["T"]),
        "Ttilde": float(first["Ttilde"]),
        "alpha": float(first["alpha"]),
        "num_samples": len(rows),
        "D1_model1_rms_abs_l2h": d1_rms,
        "D1": d1_rms,
        "Delta_init": delta_init_rms,
        "delta_scale_rms_abs_l2h": delta_rms,
        "delta_scale_mean_abs_l2h": float(np.mean(delta_values)),
        "delta_scale_std_abs_l2h": float(np.std(delta_values)),
        "delta_scale_min_abs_l2h": float(np.min(delta_values)),
        "delta_scale_max_abs_l2h": float(np.max(delta_values)),
        "Delta_scale": delta_rms,
        "Delta_dyn": delta_rms,
        "rhs_beta0": rhs_beta0_rms,
        "rhs_beta": rhs_beta_rms,
        "rhs_beta_legacy_alias_of": "rhs_beta_pathwise_rms",
        "rhs_beta_theorem_components": math.exp(float(first["beta_value"]) * float(first["T"])) * delta_init_rms
        + float(first["c_beta_T"]) * delta_rms,
        "rhs_beta_pathwise_rms": rhs_beta_rms,
        "beta_mode": first["beta_mode"],
        "beta_value": float(first["beta_value"]),
        "beta_empirical": float(first["beta_empirical"]),
        "c_beta_T": float(first["c_beta_T"]),
        "defect_metric": "pathwise_integrated_time_scaled_generator_defect",
    }


def compute_time_scaled_defect_for_dataset(
    *,
    u0: torch.Tensor,
    target_T: torch.Tensor,
    cfg: ErrorDecompositionConfig,
    Ttilde: float | None = None,
    beta_value: float | None = None,
) -> dict[str, Any]:
    """Compute sample-wise integrated time-scaled generator defect on a provided test set."""
    if u0.ndim != 2:
        raise ValueError("u0 must have shape (N, S)")
    if target_T.ndim != 2:
        raise ValueError("target_T must have shape (N, S)")
    if tuple(u0.shape) != tuple(target_T.shape):
        raise ValueError("u0 and target_T must have the same shape")
    if u0.shape[0] <= 0 or u0.shape[1] <= 1:
        raise ValueError("u0 must contain at least one sample and two spatial points")

    ttilde = float(cfg.Ttilde_values[0] if Ttilde is None else Ttilde)
    if ttilde <= 0.0:
        raise ValueError("Ttilde must be positive")
    alpha = ttilde / float(cfg.T)
    beta = _dataset_beta_value(cfg, beta_value)

    dataset_cfg = ErrorDecompositionConfig(
        **{
            **asdict(cfg),
            "num_samples": int(u0.shape[0]),
            "nx": int(u0.shape[1]),
            "Ttilde_values": [ttilde],
            "input_scale": 1.0,
            "input_shift": 0.0,
        }
    )
    _validate_config(dataset_cfg)
    result_cfg = {
        **_config_to_jsonable(cfg),
        "num_samples": int(u0.shape[0]),
        "nx": int(u0.shape[1]),
        "Ttilde_values": [ttilde],
    }

    dtype = _resolve_dtype(dataset_cfg.dtype)
    u0_work = u0.detach().cpu().to(dtype=dtype)
    z0_work = float(cfg.input_scale) * u0_work + float(cfg.input_shift)
    target_work = target_T.detach().cpu().to(dtype=dtype)
    surrogate_states = _simulate_surrogate_trajectory(z0_work, dataset_cfg)
    r_alpha = rescaled_surrogate_states(surrogate_states, dataset_cfg, alpha=alpha)
    surrogate_ttilde = r_alpha[-1]

    step_T = int(round(dataset_cfg.T / dataset_cfg.dt))
    defects = scaled_generator_defect(r_alpha, dataset_cfg, alpha=alpha)
    defect_norms = discrete_l2_h(defects.reshape(-1, defects.shape[-1]), domain_length=dataset_cfg.domain_length).reshape(step_T + 1, -1)
    weights = make_time_quadrature_weights(step_T, dataset_cfg.dt, dataset_cfg.time_quadrature).to(
        device=defect_norms.device,
        dtype=defect_norms.dtype,
    )
    delta_scale_sq = torch.sum(weights.unsqueeze(1) * defect_norms.pow(2), dim=0)
    delta_scale = torch.sqrt(delta_scale_sq)

    d1 = discrete_l2_h(target_work - surrogate_ttilde, domain_length=dataset_cfg.domain_length)
    delta_init = discrete_l2_h(u0_work - z0_work, domain_length=dataset_cfg.domain_length)
    cbeta = c_beta_T(beta, dataset_cfg.T)
    exp_beta_t = math.exp(beta * dataset_cfg.T)
    rhs_beta0_pathwise = delta_init + math.sqrt(dataset_cfg.T) * delta_scale
    rhs_beta_pathwise = exp_beta_t * delta_init + cbeta * delta_scale

    rows: list[dict[str, Any]] = []
    for idx in range(int(u0_work.shape[0])):
        delta_scale_value = float(delta_scale[idx].item())
        rhs_beta0_value = float(rhs_beta0_pathwise[idx].item())
        rhs_beta_value = float(rhs_beta_pathwise[idx].item())
        d1_value = float(d1[idx].item())
        row = {
            "sample_index": idx,
            "T": float(dataset_cfg.T),
            "Ttilde": float(ttilde),
            "alpha": float(alpha),
            "D1_model1_abs_l2h": d1_value,
            "D1_abs_l2h": d1_value,
            "Delta_init_abs_l2h": float(delta_init[idx].item()),
            "delta_scale_pathwise_abs_l2h": delta_scale_value,
            "Delta_scale_abs_l2h": delta_scale_value,
            "Delta_dyn_abs_l2h": delta_scale_value,
            "rhs_beta0_pathwise_abs_l2h": rhs_beta0_value,
            "rhs_beta_pathwise_abs_l2h": rhs_beta_value,
            "rhs_beta0_abs_l2h": rhs_beta0_value,
            "rhs_beta_abs_l2h": rhs_beta_value,
            "beta_mode": dataset_cfg.beta_mode,
            "beta_value": float(beta),
            "beta_empirical": float(beta),
            "c_beta_T": float(cbeta),
        }
        rows.append(row)

    summary = _summary_for_dataset_rows(rows)
    return {
        "config": result_cfg,
        "theory": "time_scaled_integrated_generator_defect",
        "rows": rows,
        "summary": summary,
    }


def _compute_rows_for_ttilde(
    target_states: torch.Tensor,
    surrogate_states: torch.Tensor,
    *,
    cfg: ErrorDecompositionConfig,
    Ttilde: float,
    beta_value: float,
) -> list[dict[str, Any]]:
    alpha = float(Ttilde) / float(cfg.T)
    step_T = int(round(cfg.T / cfg.dt))
    target_T = target_states[step_T]
    r_alpha = rescaled_surrogate_states(surrogate_states, cfg, alpha=alpha)
    surrogate_ttilde = r_alpha[-1]

    defects = scaled_generator_defect(r_alpha, cfg, alpha=alpha)
    defect_norms = discrete_l2_h(defects.reshape(-1, defects.shape[-1]), domain_length=cfg.domain_length).reshape(step_T + 1, -1)
    weights = make_time_quadrature_weights(step_T, cfg.dt, cfg.time_quadrature).to(
        device=defect_norms.device,
        dtype=defect_norms.dtype,
    )
    delta_scale_sq = torch.sum(weights.unsqueeze(1) * defect_norms.pow(2), dim=0)
    Delta_scale = torch.sqrt(delta_scale_sq)

    D1 = discrete_l2_h(target_T - surrogate_ttilde, domain_length=cfg.domain_length)
    Delta_init = torch.zeros_like(D1)
    cbeta = c_beta_T(beta_value, cfg.T)
    exp_beta_t = math.exp(beta_value * cfg.T)
    rhs_beta0_pathwise = Delta_init + math.sqrt(cfg.T) * Delta_scale
    rhs_beta_pathwise = exp_beta_t * Delta_init + cbeta * Delta_scale

    rows: list[dict[str, Any]] = []
    for idx in range(target_T.shape[0]):
        rhs_beta0_value = float(rhs_beta0_pathwise[idx].item())
        rhs_beta_value = float(rhs_beta_pathwise[idx].item())
        delta_scale_value = float(Delta_scale[idx].item())
        row = {
            "sample_index": idx,
            "T": float(cfg.T),
            "Ttilde": float(Ttilde),
            "alpha": float(alpha),
            "D1_abs_l2h": float(D1[idx].item()),
            "Delta_init_abs_l2h": float(Delta_init[idx].item()),
            "Delta_scale_abs_l2h": delta_scale_value,
            "rhs_beta0_pathwise_abs_l2h": rhs_beta0_value,
            "rhs_beta_pathwise_abs_l2h": rhs_beta_value,
            "beta_mode": cfg.beta_mode,
            "beta_value": float(beta_value),
            "beta_empirical": float(beta_value),
            "c_beta_T": float(cbeta),
            # Backward-compatible aliases. Delta_dyn is now exactly Delta_scale.
            "Delta_dyn_abs_l2h": delta_scale_value,
            "rhs_beta_abs_l2h": rhs_beta_value,
            "rhs_beta0_abs_l2h": rhs_beta0_value,
        }
        rows.append(row)
    return rows


def _make_calibration_trajectories(cfg: ErrorDecompositionConfig) -> tuple[torch.Tensor, torch.Tensor]:
    if cfg.calibration_num_samples <= 0:
        raise ValueError("calibration trajectories requested without calibration samples")
    calibration_seed = cfg.seed if cfg.calibration_seed is None else cfg.calibration_seed
    u0 = make_initial_conditions(cfg, num_samples=cfg.calibration_num_samples, seed=calibration_seed)
    return _simulate_target_trajectory(u0, cfg), _simulate_surrogate_trajectory(u0, cfg)


def run_error_decomposition(
    cfg: ErrorDecompositionConfig,
    *,
    save_outputs: bool = False,
) -> dict[str, Any]:
    _validate_config(cfg)
    u0 = make_initial_conditions(cfg)
    target_states = _simulate_target_trajectory(u0, cfg)
    surrogate_states = _simulate_surrogate_trajectory(u0, cfg)

    if cfg.calibration_num_samples > 0:
        calibration_target_states, calibration_surrogate_states = _make_calibration_trajectories(cfg)
    else:
        calibration_target_states, calibration_surrogate_states = target_states, surrogate_states

    rows: list[dict[str, Any]] = []
    beta_details_by_ttilde: dict[str, dict[str, Any]] = {}
    beta_values_by_ttilde: dict[str, float] = {}
    for ttilde in cfg.Ttilde_values:
        ttilde_float = float(ttilde)
        alpha = ttilde_float / float(cfg.T)
        cal_r_alpha = rescaled_surrogate_states(calibration_surrogate_states, cfg, alpha=alpha)
        cal_target = calibration_target_states[: cal_r_alpha.shape[0]]
        beta_value, beta_details = compute_beta(
            calibration_target_states=cal_target,
            calibration_surrogate_states=cal_r_alpha,
            cfg=cfg,
        )
        key = format(ttilde_float, ".12g")
        beta_details_by_ttilde[key] = {**beta_details, "Ttilde": ttilde_float, "alpha": alpha}
        beta_values_by_ttilde[key] = beta_value
        rows.extend(
            _compute_rows_for_ttilde(
                target_states,
                surrogate_states,
                cfg=cfg,
                Ttilde=ttilde_float,
                beta_value=beta_value,
            )
        )
    summary_rows = aggregate_metric_rows(rows)
    first_key = format(float(cfg.Ttilde_values[0]), ".12g")
    result = {
        "config": _config_to_jsonable(cfg),
        "full_state_special_case": True,
        "theory": "time_scaled_model1",
        "beta_mode": cfg.beta_mode,
        "beta_value": beta_values_by_ttilde[first_key],
        "beta_empirical": beta_values_by_ttilde[first_key],
        "beta_details": beta_details_by_ttilde[first_key],
        "beta_details_by_ttilde": beta_details_by_ttilde,
        "rows": rows,
        "summary_rows": summary_rows,
    }
    if save_outputs:
        save_error_decomposition_outputs(result, cfg.out_dir)
    return result


def _write_csv(path: str, rows: list[dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _scatter_with_groups(
    rows: list[dict[str, Any]],
    *,
    x_key: str,
    y_key: str,
    xlabel: str,
    ylabel: str,
    title: str,
    out_path_no_ext: str,
) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    ttilde_values = sorted({float(row["Ttilde"]) for row in rows})
    cmap = plt.get_cmap("viridis")
    for idx, ttilde in enumerate(ttilde_values):
        color = cmap(idx / max(1, len(ttilde_values) - 1))
        subset = [row for row in rows if float(row["Ttilde"]) == ttilde]
        ax.scatter(
            [row[x_key] for row in subset],
            [row[y_key] for row in subset],
            s=20,
            alpha=0.8,
            label=f"Ttilde={ttilde:g}",
            color=color,
        )
    xy_max = max(
        max(float(row[x_key]) for row in rows),
        max(float(row[y_key]) for row in rows),
        1e-12,
    )
    ax.plot([0.0, xy_max], [0.0, xy_max], linestyle="--", linewidth=1.0, color="black", alpha=0.4)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    save_figure_all_formats(fig, out_path_no_ext)
    plt.close(fig)


def _delta_scale_plot(summary_rows: list[dict[str, Any]], out_path_no_ext: str) -> None:
    ttilde = np.asarray([row["Ttilde"] for row in summary_rows], dtype=float)
    delta_scale = np.asarray([row["Delta_scale"] for row in summary_rows], dtype=float)
    d1 = np.asarray([row["D1"] for row in summary_rows], dtype=float)

    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    ax.plot(ttilde, d1, marker="o", linewidth=1.8, label="D1")
    ax.plot(ttilde, delta_scale, marker="o", linewidth=1.6, label="Delta_scale")
    ax.set_xlabel("Ttilde")
    ax.set_ylabel("absolute discrete L2h")
    ax.set_title("Time-scaled defect vs Ttilde")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    save_figure_all_formats(fig, out_path_no_ext)
    plt.close(fig)


def _scaled_bound_plot(summary_rows: list[dict[str, Any]], out_path_no_ext: str) -> None:
    ttilde = np.asarray([row["Ttilde"] for row in summary_rows], dtype=float)
    d1 = np.asarray([row["D1"] for row in summary_rows], dtype=float)
    rhs_beta = np.asarray([row["rhs_beta"] for row in summary_rows], dtype=float)
    rhs_beta0 = np.asarray([row["rhs_beta0"] for row in summary_rows], dtype=float)

    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    ax.plot(ttilde, d1, marker="o", linewidth=1.8, label="D1")
    ax.plot(ttilde, rhs_beta, marker="o", linewidth=1.6, label="rhs_beta")
    ax.plot(ttilde, rhs_beta0, marker="o", linewidth=1.6, label="rhs_beta0")
    ax.set_xlabel("Ttilde")
    ax.set_ylabel("absolute discrete L2h")
    ax.set_title("Time-scaled bound vs Ttilde")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    save_figure_all_formats(fig, out_path_no_ext)
    plt.close(fig)


def save_error_decomposition_outputs(result: dict[str, Any], out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    rows = result["rows"]
    summary_rows = result["summary_rows"]
    if not rows:
        raise ValueError("No rows to save")

    _write_csv(os.path.join(out_dir, "per_sample_metrics.csv"), rows)
    with open(os.path.join(out_dir, "per_sample_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    _write_csv(os.path.join(out_dir, "summary_metrics.csv"), summary_rows)
    with open(os.path.join(out_dir, "summary_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "config": result["config"],
                "full_state_special_case": result["full_state_special_case"],
                "theory": result["theory"],
                "beta_mode": result["beta_mode"],
                "beta_value": result["beta_value"],
                "beta_details": result["beta_details"],
                "beta_details_by_ttilde": result["beta_details_by_ttilde"],
                "summary_rows": summary_rows,
            },
            f,
            indent=2,
        )

    _scatter_with_groups(
        rows,
        x_key="rhs_beta0_pathwise_abs_l2h",
        y_key="D1_abs_l2h",
        xlabel="rhs_beta0_pathwise",
        ylabel="D1",
        title="Time-scaled bound scatter (beta0)",
        out_path_no_ext=os.path.join(out_dir, "scaled_bound_scatter_beta0"),
    )
    _scatter_with_groups(
        rows,
        x_key="rhs_beta_pathwise_abs_l2h",
        y_key="D1_abs_l2h",
        xlabel="rhs_beta_pathwise",
        ylabel="D1",
        title="Time-scaled bound scatter",
        out_path_no_ext=os.path.join(out_dir, "scaled_bound_scatter"),
    )
    _scatter_with_groups(
        rows,
        x_key="Delta_scale_abs_l2h",
        y_key="D1_abs_l2h",
        xlabel="Delta_scale",
        ylabel="D1",
        title="Delta_scale scatter",
        out_path_no_ext=os.path.join(out_dir, "delta_scale_scatter"),
    )
    _delta_scale_plot(summary_rows, os.path.join(out_dir, "delta_scale_vs_ttilde"))
    _scaled_bound_plot(summary_rows, os.path.join(out_dir, "scaled_bound_vs_ttilde"))


__all__ = [
    "ErrorDecompositionConfig",
    "aggregate_metric_rows",
    "burgers_generator",
    "c_beta_T",
    "compute_beta",
    "compute_time_scaled_defect_for_dataset",
    "defect_burgers_reservoir",
    "defect_ks_reservoir",
    "defect_reaction_diffusion_reservoir",
    "discrete_l2_h",
    "estimate_empirical_beta",
    "generator_defect",
    "interpolate_trajectory_at_times",
    "make_initial_conditions",
    "make_time_quadrature_weights",
    "rescaled_surrogate_states",
    "run_error_decomposition",
    "save_error_decomposition_outputs",
    "scaled_defect_burgers_reservoir",
    "scaled_defect_ks_reservoir",
    "scaled_defect_reaction_diffusion_reservoir",
    "scaled_generator_defect",
    "spectral_derivatives_1d",
    "target_time_grid",
]
