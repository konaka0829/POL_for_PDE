from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from pol.burgers_spectral_1d import simulate_burgers_split_step
from pol.spectral_etdrk4_1d import simulate_burgers_etdrk4

from .initial_conditions import (
    InitialConditionCoefficients,
    evaluate_initial_conditions,
    sample_gaussian_random_field_initial_conditions,
    sample_initial_condition_coefficients,
)


@dataclass(frozen=True)
class DatasetConfig:
    total_samples: int = 1200
    ntrain: int = 1000
    nval: int = 0
    ntest: int = 200
    seed: int = 0
    data_seed: int | None = None
    nx: int = 256
    domain_length: float = 1.0
    target_nu: float = 0.05
    T: float = 1.0
    dt: float = 1e-3
    fine_dt: float = 1e-4
    solver: str = "split_step"
    dealias: bool = False
    batch_size: int = 20
    dtype: str = "float64"
    ic_type: str = "fourier"
    grf_gamma: float = 2.0
    grf_tau: float = 5.0
    grf_sigma: float = 25.0
    grf_mean: float = 0.0
    fourier_num_modes: int = 8
    fourier_amplitude: float = 0.5

    def torch_dtype(self) -> torch.dtype:
        if self.dtype == "float32":
            return torch.float32
        return torch.float64


@dataclass
class DatasetBundle:
    config: DatasetConfig
    coeffs: InitialConditionCoefficients
    u0_train: torch.Tensor
    y_train: torch.Tensor
    u0_val: torch.Tensor
    y_val: torch.Tensor
    u0_test: torch.Tensor
    y_test: torch.Tensor


def _simulate_target(
    u0: torch.Tensor,
    *,
    nu: float,
    T: float,
    dt: float,
    fine_dt: float,
    solver: str,
    dealias: bool,
    batch_size: int,
    domain_length: float,
) -> torch.Tensor:
    obs_step = int(round(T / dt))
    chunks: list[torch.Tensor] = []
    for start in range(0, u0.shape[0], batch_size):
        batch = u0[start : start + batch_size]
        if solver in {"etdrk4", "fourier_pseudospectral_etdrk4"}:
            y = simulate_burgers_etdrk4(batch, nu=nu, T=T, dt=dt, dealias=dealias, domain_length=domain_length)
        elif solver in {"split_step", "semi_implicit"}:
            states = simulate_burgers_split_step(
                batch,
                dt=dt,
                Tr=T,
                obs_steps=[obs_step],
                nu=nu,
                fine_dt=fine_dt,
                forcing=None,
                forcing_steps=None,
                dealias=dealias,
                domain_length=domain_length,
            )
            y = states[-1]
        else:
            raise ValueError(f"unsupported target solver: {solver}")
        chunks.append(y.detach().cpu())
    return torch.cat(chunks, dim=0)


def build_dataset(cfg: DatasetConfig, *, device: torch.device | None = None) -> DatasetBundle:
    if cfg.total_samples != cfg.ntrain + cfg.nval + cfg.ntest:
        raise ValueError("total_samples must equal ntrain + nval + ntest")
    if cfg.ntrain <= 0:
        raise ValueError("ntrain must be positive")
    if cfg.ntest < 0:
        raise ValueError("ntest must be nonnegative")
    if cfg.total_samples <= 0:
        raise ValueError("total_samples must be positive")

    work_device = device or torch.device("cpu")
    dtype = cfg.torch_dtype()
    data_seed = cfg.seed if cfg.data_seed is None else cfg.data_seed
    if cfg.ic_type == "grf":
        coeffs = InitialConditionCoefficients(a=torch.empty((cfg.total_samples, 0), dtype=dtype), b=torch.empty((cfg.total_samples, 0), dtype=dtype))
        u0_all = sample_gaussian_random_field_initial_conditions(
            cfg.total_samples,
            cfg.nx,
            seed=data_seed,
            gamma=cfg.grf_gamma,
            tau=cfg.grf_tau,
            sigma=cfg.grf_sigma,
            mean=cfg.grf_mean,
            device=work_device,
            dtype=dtype,
        ).cpu()
    elif cfg.ic_type == "fourier":
        coeffs = sample_initial_condition_coefficients(
            cfg.total_samples,
            seed=data_seed,
            num_modes=cfg.fourier_num_modes,
            dtype=dtype,
        )
        u0_all = evaluate_initial_conditions(
            coeffs,
            cfg.nx,
            amplitude=cfg.fourier_amplitude,
            device=work_device,
            dtype=dtype,
        ).cpu()
    else:
        raise ValueError(f"unsupported ic_type: {cfg.ic_type}")
    y_all = _simulate_target(
        u0_all.to(device=work_device, dtype=dtype),
        nu=cfg.target_nu,
        T=cfg.T,
        dt=cfg.dt,
        fine_dt=cfg.fine_dt,
        solver=cfg.solver,
        dealias=cfg.dealias,
        batch_size=cfg.batch_size,
        domain_length=cfg.domain_length,
    ).cpu()
    val_start = cfg.ntrain
    test_start = cfg.ntrain + cfg.nval
    return DatasetBundle(
        config=cfg,
        coeffs=coeffs,
        u0_train=u0_all[: cfg.ntrain],
        y_train=y_all[: cfg.ntrain],
        u0_val=u0_all[val_start:test_start],
        y_val=y_all[val_start:test_start],
        u0_test=u0_all[test_start:],
        y_test=y_all[test_start:],
    )


def save_dataset_bundle(bundle: DatasetBundle, out_file: str | Path) -> None:
    path = Path(out_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        **asdict(bundle.config),
        "nu": bundle.config.target_nu,
        "equation": "burgers",
        "target_equation": "burgers",
        "time_integrator": bundle.config.solver,
        "burgers_scheme": bundle.config.solver,
    }
    payload = {
        "config": metadata,
        "coeffs_a": bundle.coeffs.a,
        "coeffs_b": bundle.coeffs.b,
        "u0_train": bundle.u0_train,
        "y_train": bundle.y_train,
        "u0_val": bundle.u0_val,
        "y_val": bundle.y_val,
        "u0_test": bundle.u0_test,
        "y_test": bundle.y_test,
        "metadata": metadata,
    }
    torch.save(payload, path)
