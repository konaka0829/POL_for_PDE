import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .donn_layers import DiffractiveLayerRaw


def _activation(name: str):
    if name == "tanh":
        return torch.tanh
    if name == "gelu":
        return F.gelu
    raise ValueError(f"Unsupported activation: {name}")


class PointwiseMLP2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, mid_channels: int) -> None:
        super().__init__()
        self.c1 = nn.Conv2d(in_channels, mid_channels, 1)
        self.c2 = nn.Conv2d(mid_channels, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.gelu(self.c1(x))
        return self.c2(x)


class DONNOperator2d(nn.Module):
    """FNO spectral block replacement using two diffractive layers + complex mixing."""

    def __init__(
        self,
        channels: int,
        spatial_size: int,
        *,
        wavelength: float,
        pixel_size: float,
        distance: float,
        phase_init: str,
        xbar_noise_std: float = 0.0,
        prop_padding: int = 0,
        projection: str = "power",
    ) -> None:
        super().__init__()
        self.channels = channels
        self.xbar_noise_std = float(xbar_noise_std)
        self.projection = projection

        scale = 1.0 / max(1, channels * channels)
        self.mix = nn.Parameter(scale * torch.randn(channels, channels, dtype=torch.complex64))

        self.donn1 = DiffractiveLayerRaw(
            spatial_size,
            wavelength=wavelength,
            pixel_size=pixel_size,
            distance=distance,
            phase_init=phase_init,
            prop_padding=prop_padding,
        )
        self.donn2 = DiffractiveLayerRaw(
            spatial_size,
            wavelength=wavelength,
            pixel_size=pixel_size,
            distance=distance,
            phase_init=phase_init,
            prop_padding=prop_padding,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        field = x.to(torch.complex64)
        field = self.donn1(field)
        mixed = torch.einsum("bchw,co->bohw", field, self.mix)

        if self.training and self.xbar_noise_std > 0.0:
            n_re = torch.randn_like(mixed.real) * self.xbar_noise_std
            n_im = torch.randn_like(mixed.real) * self.xbar_noise_std
            mixed = mixed + torch.complex(n_re, n_im)

        out = self.donn2(mixed)

        if self.projection == "real":
            return out.real
        if self.projection == "magnitude":
            return out.abs()
        if self.projection == "power":
            return out.abs().square()
        raise ValueError(f"Unsupported projection={self.projection}")


class ONE2dDarcy(nn.Module):
    """ONE baseline for Darcy with the same input/output shape as fourier_2d.py."""

    def __init__(
        self,
        *,
        spatial_size: int,
        width: int,
        in_channels: int = 1,
        domain_padding: int = 9,
        activation: str = "tanh",
        mode: str = "stagea",
        donn_ratio: float = 1.0,
        wavelength: float,
        pixel_size: float,
        distance: float,
        phase_init: str,
        xbar_noise_std: float,
        prop_padding: int,
        donn_projection: str = "power",
    ) -> None:
        super().__init__()
        self.domain_padding = int(domain_padding)
        self.act = _activation(activation)
        self.mode = mode
        self.donn_ratio = float(donn_ratio)

        donn_size = spatial_size + self.domain_padding

        self.p = nn.Linear(in_channels + 2, width)
        self.op0 = DONNOperator2d(
            width,
            donn_size,
            wavelength=wavelength,
            pixel_size=pixel_size,
            distance=distance,
            phase_init=phase_init,
            xbar_noise_std=xbar_noise_std,
            prop_padding=prop_padding,
            projection=donn_projection,
        )
        self.op1 = DONNOperator2d(
            width,
            donn_size,
            wavelength=wavelength,
            pixel_size=pixel_size,
            distance=distance,
            phase_init=phase_init,
            xbar_noise_std=xbar_noise_std,
            prop_padding=prop_padding,
            projection=donn_projection,
        )
        self.op2 = DONNOperator2d(
            width,
            donn_size,
            wavelength=wavelength,
            pixel_size=pixel_size,
            distance=distance,
            phase_init=phase_init,
            xbar_noise_std=xbar_noise_std,
            prop_padding=prop_padding,
            projection=donn_projection,
        )
        self.op3 = DONNOperator2d(
            width,
            donn_size,
            wavelength=wavelength,
            pixel_size=pixel_size,
            distance=distance,
            phase_init=phase_init,
            xbar_noise_std=xbar_noise_std,
            prop_padding=prop_padding,
            projection=donn_projection,
        )

        self.w0 = nn.Conv2d(width, width, 1)
        self.w1 = nn.Conv2d(width, width, 1)
        self.w2 = nn.Conv2d(width, width, 1)
        self.w3 = nn.Conv2d(width, width, 1)

        if self.mode == "stagea":
            self.mlp0 = PointwiseMLP2d(width, width, width)
            self.mlp1 = PointwiseMLP2d(width, width, width)
            self.mlp2 = PointwiseMLP2d(width, width, width)
            self.mlp3 = PointwiseMLP2d(width, width, width)
        elif self.mode == "tp_compat":
            self.mlp0 = nn.Identity()
            self.mlp1 = nn.Identity()
            self.mlp2 = nn.Identity()
            self.mlp3 = nn.Identity()
        else:
            raise ValueError(f"Unsupported mode={self.mode}")

        self.q = PointwiseMLP2d(width, 1, width * 4)

    def _grid(self, shape: torch.Size, device: torch.device) -> torch.Tensor:
        b, sx, sy = shape[0], shape[1], shape[2]
        gx = torch.linspace(0, 1, sx, device=device).view(1, sx, 1, 1).repeat(b, 1, sy, 1)
        gy = torch.linspace(0, 1, sy, device=device).view(1, 1, sy, 1).repeat(b, sx, 1, 1)
        return torch.cat((gx, gy), dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        grid = self._grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.p(x)
        x = x.permute(0, 3, 1, 2)

        if self.domain_padding > 0:
            x = F.pad(x, (0, self.domain_padding, 0, self.domain_padding))

        x = self.act(self.mlp0(self.donn_ratio * self.op0(x)) + self.w0(x))
        x = self.act(self.mlp1(self.donn_ratio * self.op1(x)) + self.w1(x))
        x = self.act(self.mlp2(self.donn_ratio * self.op2(x)) + self.w2(x))
        x = self.mlp3(self.donn_ratio * self.op3(x)) + self.w3(x)

        if self.domain_padding > 0:
            x = x[..., :-self.domain_padding, :-self.domain_padding]

        x = self.q(x)
        return x.permute(0, 2, 3, 1)


class ONE2dTimeNS(nn.Module):
    """ONE baseline 1-step predictor for Navier-Stokes time rollout."""

    def __init__(
        self,
        *,
        spatial_size: int,
        input_steps: int,
        width: int,
        domain_padding: int = 0,
        activation: str = "tanh",
        mode: str = "stagea",
        donn_ratio: float = 1.0,
        wavelength: float,
        pixel_size: float,
        distance: float,
        phase_init: str,
        xbar_noise_std: float,
        prop_padding: int,
        donn_projection: str = "power",
    ) -> None:
        super().__init__()
        self.domain_padding = int(domain_padding)
        self.act = _activation(activation)
        self.mode = mode
        self.donn_ratio = float(donn_ratio)
        donn_size = spatial_size + self.domain_padding

        self.p = nn.Linear(input_steps + 2, width)
        self.op0 = DONNOperator2d(
            width,
            donn_size,
            wavelength=wavelength,
            pixel_size=pixel_size,
            distance=distance,
            phase_init=phase_init,
            xbar_noise_std=xbar_noise_std,
            prop_padding=prop_padding,
            projection=donn_projection,
        )
        self.op1 = DONNOperator2d(
            width,
            donn_size,
            wavelength=wavelength,
            pixel_size=pixel_size,
            distance=distance,
            phase_init=phase_init,
            xbar_noise_std=xbar_noise_std,
            prop_padding=prop_padding,
            projection=donn_projection,
        )
        self.op2 = DONNOperator2d(
            width,
            donn_size,
            wavelength=wavelength,
            pixel_size=pixel_size,
            distance=distance,
            phase_init=phase_init,
            xbar_noise_std=xbar_noise_std,
            prop_padding=prop_padding,
            projection=donn_projection,
        )
        self.op3 = DONNOperator2d(
            width,
            donn_size,
            wavelength=wavelength,
            pixel_size=pixel_size,
            distance=distance,
            phase_init=phase_init,
            xbar_noise_std=xbar_noise_std,
            prop_padding=prop_padding,
            projection=donn_projection,
        )

        self.w0 = nn.Conv2d(width, width, 1)
        self.w1 = nn.Conv2d(width, width, 1)
        self.w2 = nn.Conv2d(width, width, 1)
        self.w3 = nn.Conv2d(width, width, 1)

        if self.mode == "stagea":
            self.mlp0 = PointwiseMLP2d(width, width, width)
            self.mlp1 = PointwiseMLP2d(width, width, width)
            self.mlp2 = PointwiseMLP2d(width, width, width)
            self.mlp3 = PointwiseMLP2d(width, width, width)
        elif self.mode == "tp_compat":
            self.mlp0 = nn.Identity()
            self.mlp1 = nn.Identity()
            self.mlp2 = nn.Identity()
            self.mlp3 = nn.Identity()
        else:
            raise ValueError(f"Unsupported mode={self.mode}")

        self.q = PointwiseMLP2d(width, 1, width * 4)

    def _grid(self, shape: torch.Size, device: torch.device) -> torch.Tensor:
        b, sx, sy = shape[0], shape[1], shape[2]
        gx = torch.linspace(0, 1, sx, device=device).view(1, sx, 1, 1).repeat(b, 1, sy, 1)
        gy = torch.linspace(0, 1, sy, device=device).view(1, 1, sy, 1).repeat(b, sx, 1, 1)
        return torch.cat((gx, gy), dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        grid = self._grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.p(x)
        x = x.permute(0, 3, 1, 2)

        if self.domain_padding > 0:
            x = F.pad(x, (0, self.domain_padding, 0, self.domain_padding))

        x = self.act(self.mlp0(self.donn_ratio * self.op0(x)) + self.w0(x))
        x = self.act(self.mlp1(self.donn_ratio * self.op1(x)) + self.w1(x))
        x = self.act(self.mlp2(self.donn_ratio * self.op2(x)) + self.w2(x))
        x = self.mlp3(self.donn_ratio * self.op3(x)) + self.w3(x)

        if self.domain_padding > 0:
            x = x[..., :-self.domain_padding, :-self.domain_padding]

        x = self.q(x)
        return x.permute(0, 2, 3, 1)
