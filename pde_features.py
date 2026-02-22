import math
from typing import Callable

import torch
import torch.nn.functional as F


def get_activation(name: str) -> Callable[[torch.Tensor], torch.Tensor]:
    name = name.lower()
    if name == "tanh":
        return torch.tanh
    if name == "gelu":
        return F.gelu
    if name == "relu":
        return F.relu
    if name == "sin":
        return torch.sin
    raise ValueError(f"Unknown activation: {name}")


def _sample_tau(
    M: int,
    tau_dist: str,
    tau_min: float,
    tau_max: float,
    tau_exp_rate: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if tau_min < 0.0 or tau_max <= 0.0 or tau_max < tau_min:
        raise ValueError("Require 0 <= tau_min <= tau_max and tau_max > 0")

    tau_dist = tau_dist.lower()
    if tau_dist == "uniform":
        u = torch.rand(M, device=device, dtype=dtype)
        return tau_min + (tau_max - tau_min) * u

    if tau_dist == "loguniform":
        if tau_min <= 0.0:
            raise ValueError("loguniform requires tau_min > 0")
        u = torch.rand(M, device=device, dtype=dtype)
        log_min = math.log(tau_min)
        log_max = math.log(tau_max)
        return torch.exp(log_min + (log_max - log_min) * u)

    if tau_dist == "exponential":
        if tau_exp_rate <= 0.0:
            raise ValueError("exponential requires tau_exp_rate > 0")
        u = torch.rand(M, device=device, dtype=dtype)
        tau = -torch.log(torch.clamp(1.0 - u, min=1e-12)) / tau_exp_rate
        return torch.clamp(tau, min=tau_min, max=tau_max)

    raise ValueError(f"Unknown tau_dist: {tau_dist}")


def apply_heat_semigroup_1d(x: torch.Tensor, tau: torch.Tensor, nu: float) -> torch.Tensor:
    """Apply heat semigroup e^{tau * nu * Delta} on 1D periodic grid.

    Args:
        x: (B, S)
        tau: scalar tensor or (B,)
        nu: diffusion coefficient
    Returns:
        (B, S)
    """
    if x.ndim != 2:
        raise ValueError(f"Expected x shape (B,S), got {tuple(x.shape)}")
    B, S = x.shape
    if tau.ndim == 0:
        tau = tau.view(1)
    tau = tau.to(device=x.device, dtype=x.dtype)
    if tau.numel() == 1:
        tau = tau.expand(B)
    if tau.numel() != B:
        raise ValueError(f"tau size mismatch: expected 1 or {B}, got {tau.numel()}")

    k = torch.fft.rfftfreq(S, d=1.0 / S).to(device=x.device, dtype=x.dtype)
    k2 = k.pow(2).view(1, -1)
    tau_col = tau.view(B, 1)
    filt = torch.exp(-nu * tau_col * ((2.0 * math.pi) ** 2) * k2)

    x_ft = torch.fft.rfft(x, dim=-1)
    y_ft = x_ft * filt
    return torch.fft.irfft(y_ft, n=S, dim=-1)


def apply_heat_semigroup_2d(x: torch.Tensor, tau: torch.Tensor, nu: float) -> torch.Tensor:
    """Apply heat semigroup e^{tau * nu * Delta} on 2D periodic grid.

    Args:
        x: (B, S, S)
        tau: scalar tensor or (B,)
        nu: diffusion coefficient
    Returns:
        (B, S, S)
    """
    if x.ndim != 3:
        raise ValueError(f"Expected x shape (B,S,S), got {tuple(x.shape)}")
    B, Sx, Sy = x.shape
    if Sx != Sy:
        raise ValueError(f"Expected square grid, got {(Sx, Sy)}")

    if tau.ndim == 0:
        tau = tau.view(1)
    tau = tau.to(device=x.device, dtype=x.dtype)
    if tau.numel() == 1:
        tau = tau.expand(B)
    if tau.numel() != B:
        raise ValueError(f"tau size mismatch: expected 1 or {B}, got {tau.numel()}")

    kx = torch.fft.fftfreq(Sx, d=1.0 / Sx).to(device=x.device, dtype=x.dtype)
    ky = torch.fft.rfftfreq(Sy, d=1.0 / Sy).to(device=x.device, dtype=x.dtype)
    k2 = (kx[:, None].pow(2) + ky[None, :].pow(2)).unsqueeze(0)
    tau_col = tau.view(B, 1, 1)
    filt = torch.exp(-nu * tau_col * ((2.0 * math.pi) ** 2) * k2)

    x_ft = torch.fft.rfft2(x, dim=(-2, -1))
    y_ft = x_ft * filt
    return torch.fft.irfft2(y_ft, s=(Sx, Sy), dim=(-2, -1))


class PDERandomFeatureMap1D:
    def __init__(
        self,
        S: int,
        M: int,
        nu: float = 1.0,
        tau_dist: str = "loguniform",
        tau_min: float = 1e-4,
        tau_max: float = 1.0,
        tau_exp_rate: float = 1.0,
        g_smooth_tau: float = 0.0,
        activation: str = "tanh",
        feature_scale: str = "inv_sqrt_m",
        inner_product: str = "mean",
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.S = S
        self.M = M
        self.nu = nu
        self.feature_scale = feature_scale
        self.inner_product = inner_product
        self.device = torch.device(device)
        self.dtype = dtype
        self.activation_name = activation
        self.activation = get_activation(activation)

        self.tau = _sample_tau(
            M=M,
            tau_dist=tau_dist,
            tau_min=tau_min,
            tau_max=tau_max,
            tau_exp_rate=tau_exp_rate,
            device=self.device,
            dtype=self.dtype,
        )

        g = torch.randn(M, S, device=self.device, dtype=self.dtype)
        if g_smooth_tau > 0.0:
            g = apply_heat_semigroup_1d(g, torch.full((M,), g_smooth_tau, device=self.device, dtype=self.dtype), nu)

        # Adjoint trick for self-adjoint heat semigroup.
        self.h = apply_heat_semigroup_1d(g, self.tau, nu)

    def features(self, a_batch: torch.Tensor) -> torch.Tensor:
        if a_batch.ndim != 2:
            raise ValueError(f"Expected a_batch shape (B,S), got {tuple(a_batch.shape)}")
        a = a_batch.to(device=self.device, dtype=self.dtype)
        z = a @ self.h.t()
        if self.inner_product == "mean":
            z = z / float(self.S)
        elif self.inner_product != "sum":
            raise ValueError(f"Unknown inner_product: {self.inner_product}")

        phi = self.activation(z)
        if self.feature_scale == "inv_sqrt_m":
            phi = phi / math.sqrt(self.M)
        elif self.feature_scale != "none":
            raise ValueError(f"Unknown feature_scale: {self.feature_scale}")
        return phi


class PDERandomFeatureMap2D:
    def __init__(
        self,
        S: int,
        M: int,
        nu: float = 1.0,
        tau_dist: str = "loguniform",
        tau_min: float = 1e-4,
        tau_max: float = 1.0,
        tau_exp_rate: float = 1.0,
        g_smooth_tau: float = 0.0,
        activation: str = "tanh",
        feature_scale: str = "inv_sqrt_m",
        inner_product: str = "mean",
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.S = S
        self.M = M
        self.nu = nu
        self.feature_scale = feature_scale
        self.inner_product = inner_product
        self.device = torch.device(device)
        self.dtype = dtype
        self.activation_name = activation
        self.activation = get_activation(activation)

        self.tau = _sample_tau(
            M=M,
            tau_dist=tau_dist,
            tau_min=tau_min,
            tau_max=tau_max,
            tau_exp_rate=tau_exp_rate,
            device=self.device,
            dtype=self.dtype,
        )

        g = torch.randn(M, S, S, device=self.device, dtype=self.dtype)
        if g_smooth_tau > 0.0:
            g = apply_heat_semigroup_2d(g, torch.full((M,), g_smooth_tau, device=self.device, dtype=self.dtype), nu)

        # Adjoint trick for self-adjoint heat semigroup.
        self.h = apply_heat_semigroup_2d(g, self.tau, nu)
        self.h_flat = self.h.reshape(M, S * S)

    def features(self, a_batch: torch.Tensor) -> torch.Tensor:
        if a_batch.ndim != 3:
            raise ValueError(f"Expected a_batch shape (B,S,S), got {tuple(a_batch.shape)}")
        B, Sx, Sy = a_batch.shape
        if Sx != self.S or Sy != self.S:
            raise ValueError(f"Expected shape (B,{self.S},{self.S}), got {tuple(a_batch.shape)}")

        a = a_batch.to(device=self.device, dtype=self.dtype).reshape(B, self.S * self.S)
        z = a @ self.h_flat.t()
        if self.inner_product == "mean":
            z = z / float(self.S * self.S)
        elif self.inner_product != "sum":
            raise ValueError(f"Unknown inner_product: {self.inner_product}")

        phi = self.activation(z)
        if self.feature_scale == "inv_sqrt_m":
            phi = phi / math.sqrt(self.M)
        elif self.feature_scale != "none":
            raise ValueError(f"Unknown feature_scale: {self.feature_scale}")
        return phi
