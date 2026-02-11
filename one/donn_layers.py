import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _to_complex(x: torch.Tensor) -> torch.Tensor:
    if torch.is_complex(x):
        return x
    return x.to(torch.float32).to(torch.complex64)


def _phase_to_raw(phi: torch.Tensor) -> torch.Tensor:
    u = (phi / (2.0 * math.pi)).clamp(1e-6, 1.0 - 1e-6)
    return torch.log(u) - torch.log1p(-u)


class DiffractiveLayerRaw(nn.Module):
    """
    Fresnel diffraction + phase-only modulation.

    Notes:
    - This is a clean re-implementation inspired by public DONN-style references,
      not a copy of third-party source code.
    - Phase mask is constrained by phi = 2*pi*sigmoid(raw_phase).
    """

    def __init__(
        self,
        size: int,
        *,
        wavelength: float,
        pixel_size: float,
        distance: float,
        phase_init: str = "uniform",
        prop_padding: int = 0,
    ) -> None:
        super().__init__()
        self.size = int(size)
        self.wavelength = float(wavelength)
        self.pixel_size = float(pixel_size)
        self.distance = float(distance)
        self.prop_padding = int(prop_padding)

        raw_phase = self._init_raw_phase(self.size, phase_init)
        self.raw_phase = nn.Parameter(raw_phase)

        self._transfer_cache: dict[tuple[torch.device, torch.dtype], torch.Tensor] = {}

    def _init_raw_phase(self, size: int, phase_init: str) -> torch.Tensor:
        if phase_init == "uniform":
            phi = 2.0 * math.pi * torch.rand(size, size, dtype=torch.float32)
            return _phase_to_raw(phi)
        if phase_init == "zero":
            return torch.zeros(size, size, dtype=torch.float32)
        if phase_init == "normal":
            return 0.05 * torch.randn(size, size, dtype=torch.float32)
        raise ValueError(f"Unsupported phase_init={phase_init}")

    def _get_transfer(self, *, device: torch.device, ctype: torch.dtype) -> torch.Tensor:
        key = (device, ctype)
        cached = self._transfer_cache.get(key)
        if cached is not None:
            return cached

        n = self.size + 2 * self.prop_padding
        rtype = torch.float32 if ctype == torch.complex64 else torch.float64
        fx = torch.fft.fftfreq(n, d=self.pixel_size, device=device, dtype=rtype)
        fy = torch.fft.fftfreq(n, d=self.pixel_size, device=device, dtype=rtype)
        fy_grid, fx_grid = torch.meshgrid(fy, fx, indexing="ij")
        phase = -math.pi * self.wavelength * self.distance * (fx_grid.square() + fy_grid.square())
        H = torch.exp(1j * phase).to(ctype)
        self._transfer_cache[key] = H
        return H

    def forward(self, field: torch.Tensor) -> torch.Tensor:
        field = _to_complex(field)
        if field.ndim != 4:
            raise ValueError(f"Expected (B,C,S,S), got shape={tuple(field.shape)}")

        if field.shape[-1] != self.size or field.shape[-2] != self.size:
            raise ValueError(
                f"Diffractive layer size mismatch: expected spatial {self.size}x{self.size}, got {field.shape[-2:]}"
            )

        if self.prop_padding > 0:
            field = F.pad(field, (self.prop_padding, self.prop_padding, self.prop_padding, self.prop_padding))

        H = self._get_transfer(device=field.device, ctype=field.dtype)
        propagated = torch.fft.ifft2(torch.fft.fft2(field) * H)

        if self.prop_padding > 0:
            p = self.prop_padding
            propagated = propagated[..., p:-p, p:-p]

        phi = 2.0 * math.pi * torch.sigmoid(self.raw_phase)
        mask = torch.exp(1j * phi).to(propagated.dtype)
        return propagated * mask.unsqueeze(0).unsqueeze(0)
