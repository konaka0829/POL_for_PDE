from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import torch


KeyType = Tuple[int, str, str]


@dataclass
class ReservoirConfig:
    reservoir: str = "reaction_diffusion"
    rd_nu: float = 1e-3
    rd_alpha: float = 1.0
    rd_beta: float = 1.0
    res_burgers_nu: float = 5e-2
    ks_nl: float = 1.0
    ks_c2: float = 1.0
    ks_c4: float = 1.0
    ks_dealias: bool = False


class Reservoir1DSolver:
    """1D periodic reservoir PDE solver with spectral derivatives."""

    def __init__(self, config: ReservoirConfig):
        self.config = config
        self._k_cache: Dict[KeyType, torch.Tensor] = {}
        self._mask_cache: Dict[KeyType, torch.Tensor] = {}

    def _cache_key(self, s: int, device: torch.device, dtype: torch.dtype) -> KeyType:
        return (s, str(device), str(dtype))

    def _wavenumbers(self, s: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        key = self._cache_key(s, device, dtype)
        if key not in self._k_cache:
            dx = 1.0 / float(s)
            k = 2.0 * torch.pi * torch.fft.rfftfreq(s, d=dx, device=device)
            self._k_cache[key] = k.to(dtype=dtype)
        return self._k_cache[key]

    def _dealias_mask(self, s: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        key = self._cache_key(s, device, dtype)
        if key not in self._mask_cache:
            cutoff = s // 3
            freqs = torch.arange(s // 2 + 1, device=device)
            mask = (freqs <= cutoff).to(dtype=dtype)
            self._mask_cache[key] = mask
        return self._mask_cache[key]

    def _apply_dealias(self, u_hat: torch.Tensor) -> torch.Tensor:
        if not self.config.ks_dealias:
            return u_hat
        s = (u_hat.shape[-1] - 1) * 2
        mask = self._dealias_mask(s, u_hat.device, u_hat.real.dtype)
        return u_hat * mask.unsqueeze(0)

    def _ux(self, z: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        z_hat = torch.fft.rfft(z, dim=-1)
        ux_hat = (1j * k) * z_hat
        return torch.fft.irfft(ux_hat, n=z.shape[-1], dim=-1)

    def _step_reaction_diffusion(self, z: torch.Tensor, dt: float, k: torch.Tensor) -> torch.Tensor:
        nu = self.config.rd_nu
        alpha = self.config.rd_alpha
        beta = self.config.rd_beta

        z_hat = torch.fft.rfft(z, dim=-1)
        nonlinear = alpha * z - beta * z.pow(3)
        rhs_hat = z_hat + dt * torch.fft.rfft(nonlinear, dim=-1)
        denom = 1.0 + dt * nu * (k.pow(2))
        next_hat = rhs_hat / denom
        return torch.fft.irfft(next_hat, n=z.shape[-1], dim=-1)

    def _step_burgers(self, z: torch.Tensor, dt: float, k: torch.Tensor) -> torch.Tensor:
        nu = self.config.res_burgers_nu
        z_hat = torch.fft.rfft(z, dim=-1)
        zx = self._ux(z, k)
        nonlinear = -z * zx
        rhs_hat = z_hat + dt * torch.fft.rfft(nonlinear, dim=-1)
        denom = 1.0 + dt * nu * (k.pow(2))
        next_hat = rhs_hat / denom
        return torch.fft.irfft(next_hat, n=z.shape[-1], dim=-1)

    def _step_ks(self, z: torch.Tensor, dt: float, k: torch.Tensor) -> torch.Tensor:
        z_hat = torch.fft.rfft(z, dim=-1)
        zx = self._ux(z, k)
        nonlinear = -self.config.ks_nl * z * zx
        n_hat = torch.fft.rfft(nonlinear, dim=-1)
        n_hat = self._apply_dealias(n_hat)

        l_hat = self.config.ks_c2 * k.pow(2) - self.config.ks_c4 * k.pow(4)
        denom = 1.0 - dt * l_hat
        next_hat = (z_hat + dt * n_hat) / denom
        next_hat = self._apply_dealias(next_hat)
        return torch.fft.irfft(next_hat, n=z.shape[-1], dim=-1)

    @torch.no_grad()
    def simulate(
        self,
        z0: torch.Tensor,
        dt: float,
        Tr: float,
        obs_steps: Iterable[int],
    ) -> List[torch.Tensor]:
        if z0.ndim != 2:
            raise ValueError(f"z0 must have shape (B, s), got {tuple(z0.shape)}")
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        if Tr <= 0.0:
            raise ValueError("Tr must be positive")

        obs_sorted = sorted(set(int(s) for s in obs_steps))
        if not obs_sorted:
            raise ValueError("obs_steps must be non-empty")
        if obs_sorted[0] < 1:
            raise ValueError("obs_steps must be >= 1")

        z = z0
        s = z.shape[-1]
        k = self._wavenumbers(s, z.device, z.dtype)

        t_steps = int(round(Tr / dt))
        if t_steps < obs_sorted[-1]:
            raise ValueError(
                f"obs step {obs_sorted[-1]} exceeds total integration steps {t_steps}"
            )

        observed: List[torch.Tensor] = []
        obs_ptr = 0
        max_obs = obs_sorted[-1]

        for step in range(1, max_obs + 1):
            if self.config.reservoir == "reaction_diffusion":
                z = self._step_reaction_diffusion(z, dt, k)
            elif self.config.reservoir == "ks":
                z = self._step_ks(z, dt, k)
            elif self.config.reservoir == "burgers":
                z = self._step_burgers(z, dt, k)
            else:
                raise ValueError(f"Unsupported reservoir: {self.config.reservoir}")

            while obs_ptr < len(obs_sorted) and step == obs_sorted[obs_ptr]:
                observed.append(z.clone())
                obs_ptr += 1

        return observed
