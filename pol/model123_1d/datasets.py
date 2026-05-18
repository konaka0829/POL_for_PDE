from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from pol.burgers_spectral_1d import simulate_burgers_split_step

from .initial_conditions import (
    InitialConditionCoefficients,
    evaluate_initial_conditions,
    sample_initial_condition_coefficients,
)


@dataclass(frozen=True)
class DatasetConfig:
    total_samples: int = 1200
    ntrain: int = 1000
    ntest: int = 200
    seed: int = 0
    nx: int = 256
    target_nu: float = 0.05
    T: float = 1.0
    dt: float = 1e-3
    fine_dt: float = 1e-4
    batch_size: int = 20
    dtype: str = "float64"

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
    u0_test: torch.Tensor
    y_test: torch.Tensor


def _simulate_target(
    u0: torch.Tensor,
    *,
    nu: float,
    T: float,
    dt: float,
    fine_dt: float,
    batch_size: int,
) -> torch.Tensor:
    obs_step = int(round(T / dt))
    chunks: list[torch.Tensor] = []
    for start in range(0, u0.shape[0], batch_size):
        batch = u0[start : start + batch_size]
        states = simulate_burgers_split_step(
            batch,
            dt=dt,
            Tr=T,
            obs_steps=[obs_step],
            nu=nu,
            fine_dt=fine_dt,
            forcing=None,
            forcing_steps=None,
            dealias=False,
        )
        chunks.append(states[-1].detach().cpu())
    return torch.cat(chunks, dim=0)


def build_dataset(cfg: DatasetConfig, *, device: torch.device | None = None) -> DatasetBundle:
    if cfg.total_samples != cfg.ntrain + cfg.ntest:
        raise ValueError("total_samples must equal ntrain + ntest")
    if cfg.ntrain <= 0:
        raise ValueError("ntrain must be positive")
    if cfg.ntest < 0:
        raise ValueError("ntest must be nonnegative")
    if cfg.total_samples <= 0:
        raise ValueError("total_samples must be positive")

    work_device = device or torch.device("cpu")
    dtype = cfg.torch_dtype()
    coeffs = sample_initial_condition_coefficients(cfg.total_samples, seed=cfg.seed, dtype=dtype)
    u0_all = evaluate_initial_conditions(coeffs, cfg.nx, device=work_device, dtype=dtype).cpu()
    y_all = _simulate_target(
        u0_all.to(device=work_device, dtype=dtype),
        nu=cfg.target_nu,
        T=cfg.T,
        dt=cfg.dt,
        fine_dt=cfg.fine_dt,
        batch_size=cfg.batch_size,
    ).cpu()
    return DatasetBundle(
        config=cfg,
        coeffs=coeffs,
        u0_train=u0_all[: cfg.ntrain],
        y_train=y_all[: cfg.ntrain],
        u0_test=u0_all[cfg.ntrain :],
        y_test=y_all[cfg.ntrain :],
    )


def save_dataset_bundle(bundle: DatasetBundle, out_file: str | Path) -> None:
    path = Path(out_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": asdict(bundle.config),
        "coeffs_a": bundle.coeffs.a,
        "coeffs_b": bundle.coeffs.b,
        "u0_train": bundle.u0_train,
        "y_train": bundle.y_train,
        "u0_test": bundle.u0_test,
        "y_test": bundle.y_test,
    }
    torch.save(payload, path)
