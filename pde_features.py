import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_activation(name: str):
    name = name.lower()
    if name == "tanh":
        return torch.tanh
    if name == "gelu":
        return F.gelu
    if name == "relu":
        return F.relu
    if name == "sin":
        return torch.sin
    raise ValueError(f"Unsupported activation: {name}")


def _sample_tau(
    m: int,
    tau_dist: str,
    tau_min: float,
    tau_max: float,
    tau_exp_rate: float,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if tau_min < 0 or tau_max <= 0 or tau_min >= tau_max:
        raise ValueError("Require 0 <= tau_min < tau_max")

    tau_dist = tau_dist.lower()
    if tau_dist == "uniform":
        u = torch.rand(m, device=device, dtype=dtype)
        return tau_min + (tau_max - tau_min) * u

    if tau_dist == "loguniform":
        if tau_min <= 0:
            raise ValueError("tau_min must be > 0 for loguniform")
        log_min = math.log(tau_min)
        log_max = math.log(tau_max)
        u = torch.rand(m, device=device, dtype=dtype)
        return torch.exp(log_min + (log_max - log_min) * u)

    if tau_dist == "exponential":
        if tau_exp_rate <= 0:
            raise ValueError("tau_exp_rate must be > 0 for exponential")
        u = torch.rand(m, device=device, dtype=dtype)
        tau = -torch.log1p(-u) / tau_exp_rate
        return torch.clamp(tau, min=tau_min, max=tau_max)

    raise ValueError(f"Unsupported tau_dist: {tau_dist}")


def _k_squared_1d(size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    k = torch.fft.rfftfreq(size, d=1.0 / size, device=device)
    return (k.to(dtype=dtype) ** 2)


def _k_squared_2d(size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    kx = torch.fft.fftfreq(size, d=1.0 / size, device=device).to(dtype=dtype)
    ky = torch.fft.rfftfreq(size, d=1.0 / size, device=device).to(dtype=dtype)
    return kx[:, None] ** 2 + ky[None, :] ** 2


def apply_heat_semigroup_1d(x: torch.Tensor, tau: torch.Tensor | float, nu: float) -> torch.Tensor:
    if x.ndim < 1:
        raise ValueError("x must have at least 1 dimension")
    size = x.shape[-1]
    k2 = _k_squared_1d(size, x.device, x.dtype)

    tau_t = torch.as_tensor(tau, device=x.device, dtype=x.dtype)
    x_ft = torch.fft.rfft(x, dim=-1)
    decay = torch.exp(-nu * tau_t[..., None] * ((2.0 * math.pi) ** 2) * k2)
    y_ft = x_ft * decay
    return torch.fft.irfft(y_ft, n=size, dim=-1)


def apply_heat_semigroup_2d(x: torch.Tensor, tau: torch.Tensor | float, nu: float) -> torch.Tensor:
    if x.ndim < 2:
        raise ValueError("x must have at least 2 dimensions")
    sx, sy = x.shape[-2], x.shape[-1]
    if sx != sy:
        raise ValueError("Only square grids are supported")
    size = sx
    k2 = _k_squared_2d(size, x.device, x.dtype)

    tau_t = torch.as_tensor(tau, device=x.device, dtype=x.dtype)
    x_ft = torch.fft.rfft2(x, dim=(-2, -1))
    decay = torch.exp(-nu * tau_t[..., None, None] * ((2.0 * math.pi) ** 2) * k2)
    y_ft = x_ft * decay
    return torch.fft.irfft2(y_ft, s=(size, size), dim=(-2, -1))


class PDERandomFeatureMap1D(nn.Module):
    def __init__(
        self,
        size: int,
        m: int,
        nu: float,
        tau_dist: str,
        tau_min: float,
        tau_max: float,
        tau_exp_rate: float,
        g_smooth_tau: float,
        activation: str,
        feature_scale: str,
        inner_product: str,
        *,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.size = int(size)
        self.m = int(m)
        self.nu = float(nu)
        self.feature_scale = feature_scale
        self.inner_product = inner_product
        self.activation_name = activation
        self.act = get_activation(activation)

        if self.m <= 0:
            raise ValueError("m must be positive")
        if self.inner_product not in {"mean", "sum"}:
            raise ValueError("inner_product must be one of {'mean', 'sum'}")
        if self.feature_scale not in {"none", "inv_sqrt_m"}:
            raise ValueError("feature_scale must be one of {'none', 'inv_sqrt_m'}")

        if device is None:
            device = torch.device("cpu")

        tau = _sample_tau(
            self.m,
            tau_dist=tau_dist,
            tau_min=tau_min,
            tau_max=tau_max,
            tau_exp_rate=tau_exp_rate,
            device=device,
            dtype=dtype,
        )
        g = torch.randn(self.m, self.size, device=device, dtype=dtype)
        if g_smooth_tau > 0:
            g = apply_heat_semigroup_1d(g, float(g_smooth_tau), self.nu)
        h = apply_heat_semigroup_1d(g, tau, self.nu)

        self.register_buffer("tau", tau)
        self.register_buffer("h", h)

    @torch.no_grad()
    def features(self, a_batch: torch.Tensor) -> torch.Tensor:
        if a_batch.ndim != 2:
            raise ValueError(f"Expected a_batch shape (B,S), got {tuple(a_batch.shape)}")
        if a_batch.shape[1] != self.size:
            raise ValueError(f"Expected S={self.size}, got {a_batch.shape[1]}")

        inner = a_batch @ self.h.T
        if self.inner_product == "mean":
            inner = inner / float(self.size)

        phi = self.act(inner)
        if self.feature_scale == "inv_sqrt_m":
            phi = phi / math.sqrt(self.m)
        return phi


class PDERandomFeatureMap2D(nn.Module):
    def __init__(
        self,
        size: int,
        m: int,
        nu: float,
        tau_dist: str,
        tau_min: float,
        tau_max: float,
        tau_exp_rate: float,
        g_smooth_tau: float,
        activation: str,
        feature_scale: str,
        inner_product: str,
        *,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.size = int(size)
        self.m = int(m)
        self.nu = float(nu)
        self.feature_scale = feature_scale
        self.inner_product = inner_product
        self.activation_name = activation
        self.act = get_activation(activation)

        if self.m <= 0:
            raise ValueError("m must be positive")
        if self.inner_product not in {"mean", "sum"}:
            raise ValueError("inner_product must be one of {'mean', 'sum'}")
        if self.feature_scale not in {"none", "inv_sqrt_m"}:
            raise ValueError("feature_scale must be one of {'none', 'inv_sqrt_m'}")

        if device is None:
            device = torch.device("cpu")

        tau = _sample_tau(
            self.m,
            tau_dist=tau_dist,
            tau_min=tau_min,
            tau_max=tau_max,
            tau_exp_rate=tau_exp_rate,
            device=device,
            dtype=dtype,
        )
        g = torch.randn(self.m, self.size, self.size, device=device, dtype=dtype)
        if g_smooth_tau > 0:
            g = apply_heat_semigroup_2d(g, float(g_smooth_tau), self.nu)
        h = apply_heat_semigroup_2d(g, tau, self.nu)

        self.register_buffer("tau", tau)
        self.register_buffer("h", h)
        self.register_buffer("h_flat", h.reshape(self.m, -1))

    @torch.no_grad()
    def features(self, a_batch: torch.Tensor) -> torch.Tensor:
        if a_batch.ndim != 3:
            raise ValueError(f"Expected a_batch shape (B,S,S), got {tuple(a_batch.shape)}")
        if a_batch.shape[1] != self.size or a_batch.shape[2] != self.size:
            raise ValueError(f"Expected (B,{self.size},{self.size}), got {tuple(a_batch.shape)}")

        a_flat = a_batch.reshape(a_batch.shape[0], -1)
        inner = a_flat @ self.h_flat.T
        if self.inner_product == "mean":
            inner = inner / float(self.size * self.size)

        phi = self.act(inner)
        if self.feature_scale == "inv_sqrt_m":
            phi = phi / math.sqrt(self.m)
        return phi


def to_torch_dtype(name: str) -> torch.dtype:
    name = name.lower()
    if name == "float32":
        return torch.float32
    if name == "float64":
        return torch.float64
    raise ValueError(f"Unsupported dtype: {name}")


def set_random_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
