import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class _PositiveNuMixin:
    def _init_nu(self, nu: float, learnable_nu: bool) -> None:
        nu = float(nu)
        if nu <= 0.0:
            raise ValueError(f"nu must be positive, got {nu}")
        self.learnable_nu = bool(learnable_nu)
        if self.learnable_nu:
            raw = math.log(math.expm1(nu))
            self.raw_nu = nn.Parameter(torch.tensor(raw, dtype=torch.float32))
        else:
            self.register_buffer("fixed_nu", torch.tensor(nu, dtype=torch.float32))

    def nu_value(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if self.learnable_nu:
            return F.softplus(self.raw_nu).to(device=device, dtype=dtype)
        return self.fixed_nu.to(device=device, dtype=dtype)


class HeatSemigroup1d(nn.Module, _PositiveNuMixin):
    """Apply heat semigroup exp(t * nu * Delta) to z with shape (B, X, C)."""

    def __init__(
        self,
        nu: float,
        learnable_nu: bool = False,
        domain_length: float = 1.0,
        use_2pi: bool = True,
    ) -> None:
        super().__init__()
        if domain_length <= 0.0:
            raise ValueError(f"domain_length must be positive, got {domain_length}")
        self.domain_length = float(domain_length)
        self.use_2pi = bool(use_2pi)
        self._k2_cache: Dict[Tuple[int, str, str], torch.Tensor] = {}
        self._init_nu(nu, learnable_nu)

    def _get_k2(self, n: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        key = (n, str(device), str(dtype))
        if key in self._k2_cache:
            return self._k2_cache[key]

        dx = self.domain_length / n
        freq = torch.fft.rfftfreq(n, d=dx).to(device=device, dtype=dtype)
        if self.use_2pi:
            k = 2.0 * math.pi * freq
        else:
            k = freq
        k2 = k.square()
        self._k2_cache[key] = k2
        return k2

    def forward(self, z: torch.Tensor, dt: float | torch.Tensor) -> torch.Tensor:
        if z.ndim != 3:
            raise ValueError(f"HeatSemigroup1d expects (B, X, C), got {tuple(z.shape)}")

        b, n, c = z.shape
        z_bc = z.permute(0, 2, 1)  # (B, C, X)
        z_ft = torch.fft.rfft(z_bc, dim=-1)

        k2 = self._get_k2(n, z.device, z.real.dtype)
        nu = self.nu_value(z.device, k2.dtype)
        dt_tensor = torch.as_tensor(dt, device=z.device, dtype=k2.dtype)
        decay = torch.exp(-nu * k2 * dt_tensor).view(1, 1, -1)

        out_ft = z_ft * decay
        out = torch.fft.irfft(out_ft, n=n, dim=-1)
        return out.permute(0, 2, 1).reshape(b, n, c)

    def apply(self, z: torch.Tensor, t: float | torch.Tensor) -> torch.Tensor:
        return self.forward(z, t)


class HeatSemigroup2d(nn.Module, _PositiveNuMixin):
    """Apply heat semigroup exp(t * nu * Delta) to z with shape (B, S, S, C)."""

    def __init__(
        self,
        nu: float,
        learnable_nu: bool = False,
        domain_length: float = 1.0,
        use_2pi: bool = True,
    ) -> None:
        super().__init__()
        if domain_length <= 0.0:
            raise ValueError(f"domain_length must be positive, got {domain_length}")
        self.domain_length = float(domain_length)
        self.use_2pi = bool(use_2pi)
        self._k2_cache: Dict[Tuple[int, str, str], torch.Tensor] = {}
        self._init_nu(nu, learnable_nu)

    def _get_k2(self, n: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        key = (n, str(device), str(dtype))
        if key in self._k2_cache:
            return self._k2_cache[key]

        dx = self.domain_length / n
        fx = torch.fft.fftfreq(n, d=dx).to(device=device, dtype=dtype)
        fy = torch.fft.rfftfreq(n, d=dx).to(device=device, dtype=dtype)

        if self.use_2pi:
            kx = 2.0 * math.pi * fx
            ky = 2.0 * math.pi * fy
        else:
            kx = fx
            ky = fy

        k2 = kx[:, None].square() + ky[None, :].square()
        self._k2_cache[key] = k2
        return k2

    def forward(self, z: torch.Tensor, dt: float | torch.Tensor) -> torch.Tensor:
        if z.ndim != 4:
            raise ValueError(f"HeatSemigroup2d expects (B, S, S, C), got {tuple(z.shape)}")
        if z.shape[1] != z.shape[2]:
            raise ValueError(f"HeatSemigroup2d expects square grid, got {tuple(z.shape)}")

        b, n, _, c = z.shape
        z_bc = z.permute(0, 3, 1, 2)  # (B, C, S, S)
        z_ft = torch.fft.rfft2(z_bc, dim=(-2, -1))

        k2 = self._get_k2(n, z.device, z.real.dtype)
        nu = self.nu_value(z.device, k2.dtype)
        dt_tensor = torch.as_tensor(dt, device=z.device, dtype=k2.dtype)
        decay = torch.exp(-nu * k2 * dt_tensor).view(1, 1, n, n // 2 + 1)

        out_ft = z_ft * decay
        out = torch.fft.irfft2(out_ft, s=(n, n), dim=(-2, -1))
        return out.permute(0, 2, 3, 1).reshape(b, n, n, c)

    def apply(self, z: torch.Tensor, t: float | torch.Tensor) -> torch.Tensor:
        return self.forward(z, t)
