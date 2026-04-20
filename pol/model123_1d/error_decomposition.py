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
from pol.reservoir_1d import Reservoir1DSolver, ReservoirConfig
from viz_utils import save_figure_all_formats


@dataclass(frozen=True)
class ErrorDecompositionConfig:
    num_samples: int = 64
    nx: int = 256
    seed: int = 0
    batch_size: int = 8
    target_nu: float = 0.05
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
    ks_b: float = 1.0
    ks_eta: float = 1.0
    ks_kappa: float = 1.0
    ks_dealias: bool = False
    dtype: str = "float64"
    device: str = "cpu"
    initial_condition_type: str = "fourier"
    grf_gamma: float = 2.0
    grf_tau: float = 5.0
    grf_sigma: float = 25.0
    grf_mean: float = 0.0
    beta_mode: str = "both"
    beta_max_states: int = 24
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
    if cfg.T <= 0.0:
        raise ValueError("T must be positive")
    if cfg.dt <= 0.0:
        raise ValueError("dt must be positive")
    if cfg.fine_dt <= 0.0:
        raise ValueError("fine_dt must be positive")
    if cfg.reservoir not in {"burgers", "reaction_diffusion", "ks"}:
        raise ValueError(f"Unsupported reservoir family: {cfg.reservoir}")
    if cfg.initial_condition_type not in {"fourier", "grf"}:
        raise ValueError("initial_condition_type must be 'fourier' or 'grf'")
    if cfg.beta_mode not in {"correlation", "empirical", "both"}:
        raise ValueError("beta_mode must be correlation, empirical, or both")
    if cfg.beta_max_states <= 1:
        raise ValueError("beta_max_states must be >= 2")
    if not cfg.Ttilde_values:
        raise ValueError("Ttilde_values must be non-empty")
    for value in cfg.Ttilde_values:
        if value <= 0.0:
            raise ValueError("All Ttilde values must be positive")


def make_initial_conditions(cfg: ErrorDecompositionConfig) -> torch.Tensor:
    device = _resolve_device(cfg.device)
    dtype = _resolve_dtype(cfg.dtype)
    if cfg.initial_condition_type == "fourier":
        coeffs = sample_initial_condition_coefficients(
            cfg.num_samples,
            seed=cfg.seed,
            dtype=dtype,
        )
        return evaluate_initial_conditions(coeffs, cfg.nx, device=device, dtype=dtype).cpu()
    return sample_gaussian_random_field_initial_conditions(
        cfg.num_samples,
        cfg.nx,
        seed=cfg.seed,
        gamma=cfg.grf_gamma,
        tau=cfg.grf_tau,
        sigma=cfg.grf_sigma,
        mean=cfg.grf_mean,
        device=device,
        dtype=dtype,
    ).cpu()


def _simulate_target_trajectory(u0: torch.Tensor, cfg: ErrorDecompositionConfig) -> torch.Tensor:
    total_steps = int(round(cfg.T / cfg.dt))
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
        )
        for idx, state in enumerate(states):
            states_per_step[idx].append(state.detach().cpu())
    return torch.stack([torch.cat(chunks, dim=0) for chunks in states_per_step], dim=0)


def _make_surrogate_solver(cfg: ErrorDecompositionConfig) -> Reservoir1DSolver:
    if cfg.reservoir == "burgers":
        return Reservoir1DSolver(
            ReservoirConfig(
                reservoir="burgers",
                res_burgers_nu=cfg.res_burgers_nu,
                res_burgers_b=cfg.res_burgers_b,
                burgers_scheme="split_step",
                burgers_fine_dt=cfg.fine_dt,
                burgers_dealias=False,
            )
        )
    if cfg.reservoir == "reaction_diffusion":
        return Reservoir1DSolver(
            ReservoirConfig(
                reservoir="reaction_diffusion",
                rd_nu=cfg.rd_nu,
                rd_alpha=cfg.rd_alpha,
                rd_beta=cfg.rd_beta,
            )
        )
    return Reservoir1DSolver(
        ReservoirConfig(
            reservoir="ks",
            ks_b=cfg.ks_b,
            ks_eta=cfg.ks_eta,
            ks_kappa=cfg.ks_kappa,
            ks_dealias=cfg.ks_dealias,
        )
    )


def _simulate_surrogate_trajectory(u0: torch.Tensor, cfg: ErrorDecompositionConfig) -> torch.Tensor:
    max_time = max(max(cfg.Ttilde_values), cfg.T)
    total_steps = int(round(max_time / cfg.dt))
    obs_steps = list(range(1, total_steps + 1))
    states_per_step: list[list[torch.Tensor]] = [[] for _ in obs_steps]
    work_device = _resolve_device(cfg.device)
    dtype = _resolve_dtype(cfg.dtype)
    solver = _make_surrogate_solver(cfg)
    for start in range(0, u0.shape[0], cfg.batch_size):
        batch = u0[start : start + cfg.batch_size].to(device=work_device, dtype=dtype)
        states = solver.simulate(batch, dt=cfg.dt, Tr=max_time, obs_steps=obs_steps)
        for idx, state in enumerate(states):
            states_per_step[idx].append(state.detach().cpu())
    return torch.stack([torch.cat(chunks, dim=0) for chunks in states_per_step], dim=0)


def discrete_l2_h(values: torch.Tensor) -> torch.Tensor:
    dx = 1.0 / float(values.shape[-1])
    return torch.sqrt(dx * torch.sum(values * values, dim=-1))


def discrete_inner_h(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    dx = 1.0 / float(a.shape[-1])
    return dx * torch.sum(a * b, dim=-1)


def aggregate_metric_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[float, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(float(row["Ttilde"]), []).append(row)

    summary_rows: list[dict[str, Any]] = []
    metric_names = [
        "D1_abs_l2h",
        "Delta_init_abs_l2h",
        "Delta_time_abs_l2h",
        "Delta_dyn_abs_l2h",
        "matched_time_error_abs_l2h",
        "matched_plus_time_abs_l2h",
        "rhs_beta0_abs_l2h",
        "rhs_beta_abs_l2h",
    ]
    for ttilde in sorted(grouped):
        t_rows = grouped[ttilde]
        summary_row: dict[str, Any] = {"Ttilde": ttilde}
        for name in metric_names:
            values = np.asarray([float(row[name]) for row in t_rows], dtype=float)
            summary_row[name.replace("_abs_l2h", "")] = float(np.sqrt(np.mean(values * values)))
        summary_row["beta_empirical"] = float(t_rows[0]["beta_empirical"])
        summary_row["c_beta_T"] = float(t_rows[0]["c_beta_T"])
        summary_rows.append(summary_row)
    return summary_rows


def spectral_derivatives_1d(z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    s = z.shape[-1]
    k = make_wavenumbers(s, z.device, z.dtype)
    z_hat = torch.fft.rfft(z, dim=-1)
    ux = torch.fft.irfft((1j * k) * z_hat, n=s, dim=-1)
    uxx = torch.fft.irfft(-(k.pow(2)) * z_hat, n=s, dim=-1)
    uxxxx = torch.fft.irfft((k.pow(4)) * z_hat, n=s, dim=-1)
    return ux, uxx, uxxxx


def burgers_generator(z: torch.Tensor, *, nu: float) -> torch.Tensor:
    ux, uxx, _ = spectral_derivatives_1d(z)
    return nu * uxx - z * ux


def defect_burgers_reservoir(
    z: torch.Tensor,
    *,
    target_nu: float,
    res_burgers_nu: float,
    res_burgers_b: float,
) -> torch.Tensor:
    ux, uxx, _ = spectral_derivatives_1d(z)
    return (target_nu - res_burgers_nu) * uxx + (res_burgers_b - 1.0) * z * ux


def defect_reaction_diffusion_reservoir(
    z: torch.Tensor,
    *,
    target_nu: float,
    rd_nu: float,
    rd_alpha: float,
    rd_beta: float,
) -> torch.Tensor:
    ux, uxx, _ = spectral_derivatives_1d(z)
    return (target_nu - rd_nu) * uxx - z * ux - rd_alpha * z + rd_beta * z.pow(3)


def defect_ks_reservoir(
    z: torch.Tensor,
    *,
    target_nu: float,
    ks_b: float,
    ks_eta: float,
    ks_kappa: float,
) -> torch.Tensor:
    ux, uxx, uxxxx = spectral_derivatives_1d(z)
    return (target_nu + ks_eta) * uxx + (ks_b - 1.0) * z * ux + ks_kappa * uxxxx


def generator_defect(z: torch.Tensor, cfg: ErrorDecompositionConfig) -> torch.Tensor:
    if cfg.reservoir == "burgers":
        return defect_burgers_reservoir(
            z,
            target_nu=cfg.target_nu,
            res_burgers_nu=cfg.res_burgers_nu,
            res_burgers_b=cfg.res_burgers_b,
        )
    if cfg.reservoir == "reaction_diffusion":
        return defect_reaction_diffusion_reservoir(
            z,
            target_nu=cfg.target_nu,
            rd_nu=cfg.rd_nu,
            rd_alpha=cfg.rd_alpha,
            rd_beta=cfg.rd_beta,
        )
    return defect_ks_reservoir(
        z,
        target_nu=cfg.target_nu,
        ks_b=cfg.ks_b,
        ks_eta=cfg.ks_eta,
        ks_kappa=cfg.ks_kappa,
    )


def estimate_empirical_beta(
    target_states: torch.Tensor,
    surrogate_states: torch.Tensor,
    cfg: ErrorDecompositionConfig,
) -> float:
    states = torch.cat([target_states, surrogate_states[: target_states.shape[0]]], dim=1)
    flat_states = states.reshape(-1, states.shape[-1])
    if flat_states.shape[0] > cfg.beta_max_states:
        keep = torch.linspace(
            0,
            flat_states.shape[0] - 1,
            steps=cfg.beta_max_states,
            dtype=torch.float64,
        ).round().long()
        flat_states = flat_states.index_select(0, keep)

    if flat_states.shape[0] <= 1:
        return 0.0

    F = burgers_generator(flat_states, nu=cfg.target_nu)
    beta = -float("inf")
    for i in range(flat_states.shape[0]):
        zi = flat_states[i : i + 1]
        Fi = F[i : i + 1]
        for j in range(i):
            dz = zi - flat_states[j : j + 1]
            denom = discrete_inner_h(dz, dz).item()
            if denom <= 1e-14:
                continue
            dF = Fi - F[j : j + 1]
            ratio = discrete_inner_h(dF, dz).item() / denom
            beta = max(beta, ratio)
    if beta == -float("inf"):
        return 0.0
    return float(beta)


def c_beta_T(beta: float, T: float) -> float:
    if abs(beta) < 1e-12:
        return math.sqrt(T)
    value = (math.exp(2.0 * beta * T) - 1.0) / (2.0 * beta)
    return math.sqrt(max(value, 0.0))


def _compute_rows_for_ttilde(
    target_states: torch.Tensor,
    surrogate_states: torch.Tensor,
    *,
    cfg: ErrorDecompositionConfig,
    Ttilde: float,
    beta_empirical: float,
) -> list[dict[str, Any]]:
    step_T = int(round(cfg.T / cfg.dt))
    step_ttilde = int(round(Ttilde / cfg.dt))

    target_T = target_states[step_T - 1]
    surrogate_T = surrogate_states[step_T - 1]
    surrogate_ttilde = surrogate_states[step_ttilde - 1]
    defects = generator_defect(surrogate_states[:step_T], cfg)

    D1 = discrete_l2_h(target_T - surrogate_ttilde)
    matched = discrete_l2_h(target_T - surrogate_T)
    Delta_time = discrete_l2_h(surrogate_T - surrogate_ttilde)
    defect_norms = discrete_l2_h(defects.reshape(-1, defects.shape[-1])).reshape(step_T, -1)
    Delta_dyn = torch.sqrt(cfg.dt * torch.sum(defect_norms.pow(2), dim=0))

    Delta_init = torch.zeros_like(D1)
    cbeta = c_beta_T(beta_empirical, cfg.T)
    rhs_beta0 = math.sqrt(cfg.T) * Delta_dyn + Delta_time
    rhs_beta = cbeta * Delta_dyn + Delta_time
    matched_rhs_beta = cbeta * Delta_dyn
    matched_plus_time = matched + Delta_time

    rows: list[dict[str, Any]] = []
    for idx in range(target_T.shape[0]):
        rows.append(
            {
                "sample_index": idx,
                "Ttilde": float(Ttilde),
                "D1_abs_l2h": float(D1[idx].item()),
                "Delta_init_abs_l2h": float(Delta_init[idx].item()),
                "Delta_time_abs_l2h": float(Delta_time[idx].item()),
                "Delta_dyn_abs_l2h": float(Delta_dyn[idx].item()),
                "matched_time_error_abs_l2h": float(matched[idx].item()),
                "matched_plus_time_abs_l2h": float(matched_plus_time[idx].item()),
                "rhs_beta0_abs_l2h": float(rhs_beta0[idx].item()),
                "rhs_beta_abs_l2h": float(rhs_beta[idx].item()),
                "matched_rhs_beta_abs_l2h": float(matched_rhs_beta[idx].item()),
                "beta_empirical": float(beta_empirical),
                "c_beta_T": float(cbeta),
            }
        )
    return rows


def run_error_decomposition(
    cfg: ErrorDecompositionConfig,
    *,
    save_outputs: bool = False,
) -> dict[str, Any]:
    _validate_config(cfg)
    u0 = make_initial_conditions(cfg)
    target_states = _simulate_target_trajectory(u0, cfg)
    surrogate_states = _simulate_surrogate_trajectory(u0, cfg)
    beta_emp = estimate_empirical_beta(target_states, surrogate_states, cfg)

    rows: list[dict[str, Any]] = []
    for ttilde in cfg.Ttilde_values:
        rows.extend(
            _compute_rows_for_ttilde(
                target_states,
                surrogate_states,
                cfg=cfg,
                Ttilde=float(ttilde),
                beta_empirical=beta_emp,
            )
        )
    summary_rows = aggregate_metric_rows(rows)
    result = {
        "config": _config_to_jsonable(cfg),
        "beta_empirical": beta_emp,
        "rows": rows,
        "summary_rows": summary_rows,
    }
    if save_outputs:
        save_error_decomposition_outputs(result, cfg.out_dir, cfg.beta_mode)
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


def _time_mismatch_envelope_plot(summary_rows: list[dict[str, Any]], out_path_no_ext: str) -> None:
    ttilde = np.asarray([row["Ttilde"] for row in summary_rows], dtype=float)
    matched = np.asarray([row["matched_time_error"] for row in summary_rows], dtype=float)
    delta_time = np.asarray([row["Delta_time"] for row in summary_rows], dtype=float)
    d1 = np.asarray([row["D1"] for row in summary_rows], dtype=float)
    envelope = matched + delta_time

    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    ax.plot(ttilde, d1, marker="o", linewidth=1.8, label="D1")
    ax.plot(ttilde, matched, marker="o", linewidth=1.4, label="matched-time error")
    ax.plot(ttilde, delta_time, marker="o", linewidth=1.4, label="Delta_time")
    ax.fill_between(ttilde, matched, envelope, alpha=0.2, label="matched + Delta_time")
    ax.plot(ttilde, envelope, marker="o", linewidth=1.4, linestyle="--", label="envelope upper")
    ax.set_xlabel("Ttilde")
    ax.set_ylabel("absolute discrete L2")
    ax.set_title("Time-mismatch envelope")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    save_figure_all_formats(fig, out_path_no_ext)
    plt.close(fig)


def save_error_decomposition_outputs(
    result: dict[str, Any],
    out_dir: str,
    beta_mode: str,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    rows = result["rows"]
    summary_rows = result["summary_rows"]
    if not rows:
        raise ValueError("No rows to save")

    per_sample_csv = os.path.join(out_dir, "per_sample_metrics.csv")
    _write_csv(per_sample_csv, rows)
    with open(os.path.join(out_dir, "per_sample_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    _write_csv(os.path.join(out_dir, "summary_metrics.csv"), summary_rows)
    with open(os.path.join(out_dir, "summary_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "config": result["config"],
                "beta_empirical": result["beta_empirical"],
                "summary_rows": summary_rows,
            },
            f,
            indent=2,
        )

    if beta_mode in {"correlation", "both"}:
        _scatter_with_groups(
            rows,
            x_key="Delta_dyn_abs_l2h",
            y_key="matched_time_error_abs_l2h",
            xlabel="Delta_dyn",
            ylabel="matched-time error",
            title="Matched-time scatter (correlation mode)",
            out_path_no_ext=os.path.join(out_dir, "matched_time_scatter_correlation"),
        )
        _scatter_with_groups(
            rows,
            x_key="rhs_beta0_abs_l2h",
            y_key="D1_abs_l2h",
            xlabel="sqrt(T) * Delta_dyn + Delta_time",
            ylabel="D1",
            title="Combined scatter (correlation mode)",
            out_path_no_ext=os.path.join(out_dir, "combined_scatter_correlation"),
        )

    if beta_mode in {"empirical", "both"}:
        _scatter_with_groups(
            rows,
            x_key="matched_rhs_beta_abs_l2h",
            y_key="matched_time_error_abs_l2h",
            xlabel="c_beta,T * Delta_dyn",
            ylabel="matched-time error",
            title="Matched-time scatter (empirical beta mode)",
            out_path_no_ext=os.path.join(out_dir, "matched_time_scatter_empirical"),
        )
        _scatter_with_groups(
            rows,
            x_key="rhs_beta_abs_l2h",
            y_key="D1_abs_l2h",
            xlabel="c_beta,T * Delta_dyn + Delta_time",
            ylabel="D1",
            title="Combined scatter (empirical beta mode)",
            out_path_no_ext=os.path.join(out_dir, "combined_scatter_empirical"),
        )

    _scatter_with_groups(
        rows,
        x_key="Delta_time_abs_l2h",
        y_key="D1_abs_l2h",
        xlabel="Delta_time",
        ylabel="D1",
        title="Time-mismatch scatter",
        out_path_no_ext=os.path.join(out_dir, "time_mismatch_scatter"),
    )
    _time_mismatch_envelope_plot(summary_rows, os.path.join(out_dir, "time_mismatch_envelope"))


__all__ = [
    "ErrorDecompositionConfig",
    "aggregate_metric_rows",
    "burgers_generator",
    "c_beta_T",
    "defect_burgers_reservoir",
    "defect_ks_reservoir",
    "defect_reaction_diffusion_reservoir",
    "discrete_l2_h",
    "estimate_empirical_beta",
    "generator_defect",
    "make_initial_conditions",
    "run_error_decomposition",
    "save_error_decomposition_outputs",
    "spectral_derivatives_1d",
]
