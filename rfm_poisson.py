"""Poisson solver and smoothing utilities for Darcy RFM features."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F


def dst1(x: torch.Tensor) -> torch.Tensor:
    """DST-I using FFT trick along the last dimension."""
    n = x.shape[-1]
    zeros = torch.zeros(*x.shape[:-1], 1, device=x.device, dtype=x.dtype)
    x_flip = torch.flip(x, dims=[-1])
    y = torch.cat([zeros, x, zeros, -x_flip], dim=-1)
    y_fft = torch.fft.fft(y, dim=-1)
    dst = y_fft.imag[..., 1 : n + 1]
    return dst


def idst1(x: torch.Tensor) -> torch.Tensor:
    """Inverse DST-I (up to normalization)."""
    n = x.shape[-1]
    y = dst1(x)
    return y / (2.0 * (n + 1))


def dst2(x: torch.Tensor) -> torch.Tensor:
    y = dst1(x)
    y = dst1(y.transpose(-1, -2)).transpose(-1, -2)
    return y


def idst2(x: torch.Tensor) -> torch.Tensor:
    y = idst1(x)
    y = idst1(y.transpose(-1, -2)).transpose(-1, -2)
    return y


def poisson_solve_dirichlet(rhs: torch.Tensor, s: int) -> torch.Tensor:
    """Solve -Δu = rhs on an s x s grid with zero Dirichlet boundary."""
    if rhs.ndim != 3:
        raise ValueError(f"rhs must be (batch, s, s), got {rhs.shape}")
    batch, h, w = rhs.shape
    if h != s or w != s:
        raise ValueError(f"rhs spatial size mismatch: expected ({s},{s}), got ({h},{w})")

    n = s - 2
    h_spacing = 1.0 / (s - 1)

    rhs_interior = rhs[:, 1:-1, 1:-1]
    rhs_hat = dst2(rhs_interior)

    p = torch.arange(1, n + 1, device=rhs.device, dtype=rhs.dtype)
    cos_p = torch.cos(torch.pi * p / (n + 1))
    lam = (2 - 2 * cos_p)[:, None] + (2 - 2 * cos_p)[None, :]
    lam = lam / (h_spacing**2)

    u_hat = rhs_hat / lam
    u_interior = idst2(u_hat)

    u = torch.zeros((batch, s, s), device=rhs.device, dtype=rhs.dtype)
    u[:, 1:-1, 1:-1] = u_interior
    return u


def gradient_centered(u: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Centered gradient with replicate padding (Neumann-like)."""
    if u.ndim != 3:
        raise ValueError(f"u must be (batch, s, s), got {u.shape}")
    s = u.shape[-1]
    h = 1.0 / (s - 1)
    u_pad = F.pad(u, (1, 1, 1, 1), mode="replicate")
    du_dx = (u_pad[:, 2:, 1:-1] - u_pad[:, :-2, 1:-1]) / (2 * h)
    du_dy = (u_pad[:, 1:-1, 2:] - u_pad[:, 1:-1, :-2]) / (2 * h)
    return du_dx, du_dy


def heat_smooth_neumann(v: torch.Tensor, eta: float, dt: float, steps: int) -> torch.Tensor:
    """Explicit heat equation smoothing with Neumann-like boundaries."""
    out = v
    s = v.shape[-1]
    h = 1.0 / (s - 1)
    for _ in range(steps):
        v_pad = F.pad(out, (1, 1, 1, 1), mode="replicate")
        lap = (
            v_pad[:, 2:, 1:-1]
            + v_pad[:, :-2, 1:-1]
            + v_pad[:, 1:-1, 2:]
            + v_pad[:, 1:-1, :-2]
            - 4.0 * out
        ) / (h**2)
        out = out + dt * eta * lap
    return out
