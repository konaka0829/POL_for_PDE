from __future__ import annotations

import math
import torch


def solve_heat_exact(values: torch.Tensor, *, nu: float, T: float, domain_length: float) -> torch.Tensor:
    """Apply the exact periodic heat semigroup along the last (endpoint-free) grid axis."""
    if values.ndim < 1 or values.shape[-1] < 2:
        raise ValueError("heat input must have shape (..., n), n >= 2")
    if not values.dtype.is_floating_point or values.is_complex():
        raise TypeError("heat input must be real float32 or float64")
    if values.dtype not in (torch.float32, torch.float64):
        raise TypeError("heat input must be float32 or float64")
    if not math.isfinite(nu) or nu < 0 or not math.isfinite(T) or T < 0 or not math.isfinite(domain_length) or domain_length <= 0:
        raise ValueError(f"invalid heat parameters nu={nu}, T={T}, L={domain_length}, n={values.shape[-1]}, dtype={values.dtype}, device={values.device}")
    n = values.shape[-1]
    k = torch.fft.rfftfreq(n, d=domain_length / n, device=values.device, dtype=values.dtype)
    multiplier = torch.exp(-float(nu) * float(T) * (2 * torch.pi * k) ** 2)
    result = torch.fft.irfft(torch.fft.rfft(values, dim=-1) * multiplier, n=n, dim=-1).to(values.dtype)
    if not bool(torch.isfinite(result).all()):
        raise FloatingPointError(f"non-finite heat result: nu={nu}, T={T}, L={domain_length}, n={n}, dtype={values.dtype}, device={values.device}")
    return result


def heat_multiplier_vector(q: int, *, target_nu: float, target_T: float, surrogate_nu: float, surrogate_T: float, domain_length: float, dtype: torch.dtype = torch.float64, device: torch.device | str = "cpu") -> torch.Tensor:
    if q <= 0 or q % 2 == 0:
        raise ValueError("q must be a positive odd integer")
    delta = target_nu * target_T - surrogate_nu * surrogate_T
    ks = torch.arange(1, (q - 1) // 2 + 1, dtype=dtype, device=device)
    modes = torch.exp(-delta * (2 * torch.pi * ks / domain_length) ** 2)
    return torch.cat((torch.ones(1, dtype=dtype, device=device), torch.repeat_interleave(modes, 2)))


def heat_regime(*, target_nu: float, target_T: float, surrogate_nu: float, surrogate_T: float) -> tuple[str, float]:
    delta = target_nu * target_T - surrogate_nu * surrogate_T
    if abs(delta) <= 1e-14 * max(1.0, abs(target_nu * target_T), abs(surrogate_nu * surrogate_T)):
        raise ValueError("exact-match heat case is excluded from E1")
    return ("stable" if delta > 0 else "unstable"), delta
