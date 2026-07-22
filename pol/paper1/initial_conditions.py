from __future__ import annotations

from dataclasses import dataclass

import torch

from pol.model123_1d.initial_conditions import sample_gaussian_random_field_initial_conditions

from .config import Paper1Config
from .grids import spectral_resample_periodic


@dataclass(frozen=True)
class MasterInitialConditions:
    sample_ids: torch.Tensor
    values_master: torch.Tensor
    fourier_master: torch.Tensor
    master_nx: int
    domain_length: float
    seed: int


def resolve_device(name: str) -> torch.device:
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        return torch.device("cuda")
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cpu":
        return torch.device("cpu")
    raise ValueError(f"unsupported device: {name}")


def build_master_grf_initial_conditions(config: Paper1Config) -> MasterInitialConditions:
    config.validate()
    dtype = config.data.torch_dtype()
    device = resolve_device(config.data.device)
    values = sample_gaussian_random_field_initial_conditions(
        config.data.total_samples,
        config.spatial.reference_nx,
        seed=config.data.seed,
        gamma=config.data.grf_gamma,
        tau=config.data.grf_tau,
        sigma=config.data.grf_sigma,
        mean=config.data.grf_mean,
        device=device,
        dtype=dtype,
    )
    fourier = torch.fft.rfft(values, dim=-1, norm="forward")
    return MasterInitialConditions(
        sample_ids=torch.arange(config.data.total_samples, dtype=torch.long, device=device),
        values_master=values,
        fourier_master=fourier,
        master_nx=config.spatial.reference_nx,
        domain_length=config.domain.length,
        seed=config.data.seed,
    )


def initial_conditions_at_resolution(
    master: MasterInitialConditions,
    nx: int,
) -> torch.Tensor:
    return spectral_resample_periodic(master.values_master, nx, domain_length=master.domain_length)
