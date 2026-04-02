from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class InitialConditionCoefficients:
    a: torch.Tensor
    b: torch.Tensor

    @property
    def num_samples(self) -> int:
        return int(self.a.shape[0])

    @property
    def num_modes(self) -> int:
        return int(self.a.shape[1])


def sample_initial_condition_coefficients(
    num_samples: int,
    *,
    seed: int,
    num_modes: int = 8,
    dtype: torch.dtype = torch.float64,
) -> InitialConditionCoefficients:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    a = torch.randn((num_samples, num_modes), generator=gen, dtype=dtype)
    b = torch.randn((num_samples, num_modes), generator=gen, dtype=dtype)
    return InitialConditionCoefficients(a=a, b=b)


def evaluate_initial_conditions(
    coeffs: InitialConditionCoefficients,
    nx: int,
    *,
    amplitude: float = 0.5,
    device: torch.device,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    if nx <= 1:
        raise ValueError("nx must be >= 2")
    if amplitude <= 0.0:
        raise ValueError("amplitude must be positive")

    x = torch.linspace(0.0, 1.0, steps=nx + 1, device=device, dtype=dtype)[:-1].unsqueeze(0)
    a = coeffs.a.to(device=device, dtype=dtype)
    b = coeffs.b.to(device=device, dtype=dtype)
    u = torch.zeros((coeffs.num_samples, nx), device=device, dtype=dtype)
    two_pi = 2.0 * torch.pi

    for m in range(1, coeffs.num_modes + 1):
        scale = 1.0 / float(m * m)
        phase = two_pi * float(m) * x
        u = u + scale * (a[:, m - 1 : m] * torch.cos(phase) + b[:, m - 1 : m] * torch.sin(phase))

    inf_norm = u.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12)
    return amplitude * u / inf_norm


def sample_gaussian_random_field_initial_conditions(
    num_samples: int,
    nx: int,
    *,
    seed: int,
    gamma: float = 2.0,
    tau: float = 5.0,
    sigma: float = 25.0,
    mean: float = 0.0,
    device: torch.device,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    if nx <= 1:
        raise ValueError("nx must be >= 2")
    if gamma <= 0.0:
        raise ValueError("gamma must be positive")
    if tau < 0.0:
        raise ValueError("tau must be non-negative")
    if sigma < 0.0:
        raise ValueError("sigma must be non-negative")

    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)

    xfreq = torch.fft.rfftfreq(nx, d=1.0 / float(nx), device=device).to(dtype=dtype)
    wavenumbers = 2.0 * torch.pi * xfreq
    eigvals = (sigma**2) * torch.pow(wavenumbers.pow(2) + tau**2, -gamma)
    eigvals = eigvals.to(device=device, dtype=dtype)

    coeff_shape = (num_samples, xfreq.shape[0])
    real_part = torch.zeros(coeff_shape, device=device, dtype=dtype)
    imag_part = torch.zeros(coeff_shape, device=device, dtype=dtype)

    # Match MATLAB GRF1.m periodic sampling:
    # - only positive modes k >= 1 are randomized
    # - the constant mode is deterministic and set by `mean`
    # - uu(t) is shifted to uu(t - 0.5), which multiplies Fourier mode k by (-1)^k
    real_part[:, 0] = float(mean)

    if nx % 2 == 0:
        nyquist_idx = coeff_shape[1] - 1
        nyquist_sign = -1.0 if (nx // 2) % 2 else 1.0
        real_part[:, nyquist_idx] = nyquist_sign * torch.sqrt(2.0 * eigvals[nyquist_idx]) * torch.randn(
            num_samples, generator=gen, dtype=dtype, device=device
        )
        interior_end = nyquist_idx
    else:
        interior_end = coeff_shape[1]

    n_interior = interior_end - 1
    if n_interior > 0:
        signs = torch.where(
            (torch.arange(1, interior_end, device=device) % 2) == 0,
            torch.ones(interior_end - 1, device=device, dtype=dtype),
            -torch.ones(interior_end - 1, device=device, dtype=dtype),
        )
        std = torch.sqrt(eigvals[1:interior_end] / 2.0).unsqueeze(0)
        real_noise = torch.randn((num_samples, n_interior), generator=gen, dtype=dtype, device=device)
        imag_noise = torch.randn((num_samples, n_interior), generator=gen, dtype=dtype, device=device)
        real_part[:, 1:interior_end] = signs.unsqueeze(0) * std * real_noise
        imag_part[:, 1:interior_end] = signs.unsqueeze(0) * std * imag_noise

    complex_dtype = torch.complex64 if dtype == torch.float32 else torch.complex128
    coeffs = torch.complex(real_part, imag_part).to(dtype=complex_dtype)
    return torch.fft.irfft(coeffs, n=nx, dim=-1, norm="forward")
