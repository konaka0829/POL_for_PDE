from __future__ import annotations

import math

import torch

from .grids import periodic_grid


def equispaced_periodic_positions(
    J: int,
    domain_length: float,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    if J <= 0:
        raise ValueError("J must be positive")
    if domain_length <= 0.0:
        raise ValueError("domain_length must be positive")
    return torch.arange(J, device=device, dtype=dtype) * (float(domain_length) / float(J))


def _evaluate_trig_interpolant(values: torch.Tensor, J: int) -> torch.Tensor:
    n = int(values.shape[-1])
    coeffs = torch.fft.fft(values, dim=-1, norm="forward")
    j = torch.arange(J, device=values.device, dtype=values.dtype)
    out = coeffs[..., 0:1].real.expand(*values.shape[:-1], J).clone()
    non_nyq = (n - 1) // 2
    for k in range(1, non_nyq + 1):
        phase = 2.0 * torch.pi * float(k) * j / float(J)
        c = coeffs[..., k]
        out = out + 2.0 * (c.real.unsqueeze(-1) * torch.cos(phase) - c.imag.unsqueeze(-1) * torch.sin(phase))
    if n % 2 == 0:
        phase = torch.pi * float(n) * j / float(J)
        out = out + coeffs[..., n // 2].real.unsqueeze(-1) * torch.cos(phase)
    return out


def observe_equispaced_periodic(
    values: torch.Tensor,
    J: int,
    *,
    domain_length: float,
    l2_scale: bool = True,
) -> torch.Tensor:
    """Evaluate the source-grid trigonometric interpolant at ``x_j=jL/J``."""
    if values.ndim < 1:
        raise ValueError("values must have shape (..., n_source)")
    if values.shape[-1] < 2:
        raise ValueError("n_source must be >= 2")
    if J <= 0:
        raise ValueError("J must be positive")
    if domain_length <= 0.0:
        raise ValueError("domain_length must be positive")
    n = int(values.shape[-1])
    if n % J == 0:
        idx = torch.arange(J, device=values.device, dtype=torch.long) * (n // J)
        observed = values.index_select(dim=-1, index=idx)
    else:
        observed = _evaluate_trig_interpolant(values, J)
    if l2_scale:
        observed = observed * math.sqrt(float(domain_length) / float(J))
    return observed.to(dtype=values.dtype)
