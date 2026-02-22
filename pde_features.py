import math
from typing import Optional, Tuple, Union

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


def _sample_c(
    m: int,
    dist: str,
    c_min: float,
    c_max: float,
    c_std: float,
    c_fixed: float,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    dist = dist.lower()
    if dist == "uniform":
        if c_min >= c_max:
            raise ValueError("Require c_min < c_max for uniform")
        u = torch.rand(m, device=device, dtype=dtype)
        return c_min + (c_max - c_min) * u
    if dist == "normal":
        if c_std <= 0:
            raise ValueError("c_std must be > 0 for normal")
        return c_std * torch.randn(m, device=device, dtype=dtype)
    if dist == "fixed":
        return torch.full((m,), float(c_fixed), device=device, dtype=dtype)
    raise ValueError(f"Unsupported c_dist: {dist}")


def _sample_alpha(
    m: int,
    dist: str,
    alpha_min: float,
    alpha_max: float,
    alpha_fixed: float,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    dist = dist.lower()
    if dist == "uniform":
        if alpha_min <= 0 or alpha_min >= alpha_max:
            raise ValueError("Require 0 < alpha_min < alpha_max for uniform")
        u = torch.rand(m, device=device, dtype=dtype)
        return alpha_min + (alpha_max - alpha_min) * u
    if dist == "loguniform":
        if alpha_min <= 0 or alpha_max <= 0 or alpha_min >= alpha_max:
            raise ValueError("Require 0 < alpha_min < alpha_max for loguniform")
        log_min = math.log(alpha_min)
        log_max = math.log(alpha_max)
        u = torch.rand(m, device=device, dtype=dtype)
        return torch.exp(log_min + (log_max - log_min) * u)
    if dist == "fixed":
        if alpha_fixed <= 0:
            raise ValueError("alpha_fixed must be > 0 for fixed")
        return torch.full((m,), float(alpha_fixed), device=device, dtype=dtype)
    raise ValueError(f"Unsupported alpha_dist: {dist}")


def _sample_spd_2x2(
    m: int,
    eig_dist: str,
    eig_min: float,
    eig_max: float,
    theta_dist: str,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    eig_dist = eig_dist.lower()
    if eig_dist == "uniform":
        if eig_min >= eig_max:
            raise ValueError("Require eig_min < eig_max for uniform")
        u = torch.rand(m, device=device, dtype=dtype)
        d1 = eig_min + (eig_max - eig_min) * u
        u = torch.rand(m, device=device, dtype=dtype)
        d2 = eig_min + (eig_max - eig_min) * u
    elif eig_dist == "loguniform":
        if eig_min <= 0 or eig_max <= 0 or eig_min >= eig_max:
            raise ValueError("Require 0 < eig_min < eig_max for loguniform")
        log_min = math.log(eig_min)
        log_max = math.log(eig_max)
        u = torch.rand(m, device=device, dtype=dtype)
        d1 = torch.exp(log_min + (log_max - log_min) * u)
        u = torch.rand(m, device=device, dtype=dtype)
        d2 = torch.exp(log_min + (log_max - log_min) * u)
    else:
        raise ValueError(f"Unsupported aniso eig_dist: {eig_dist}")

    theta_dist = theta_dist.lower()
    if theta_dist != "uniform":
        raise ValueError("Unsupported aniso theta_dist (only 'uniform')")
    theta = math.pi * torch.rand(m, device=device, dtype=dtype)

    c = torch.cos(theta)
    s = torch.sin(theta)
    d11 = d1 * c * c + d2 * s * s
    d22 = d1 * s * s + d2 * c * c
    d12 = (d1 - d2) * s * c
    return d11, d12, d22


def _k_squared_1d(size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    k = torch.fft.rfftfreq(size, d=1.0 / size, device=device)
    return k.to(dtype=dtype) ** 2


def _k_squared_2d(size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    kx = torch.fft.fftfreq(size, d=1.0 / size, device=device).to(dtype=dtype)
    ky = torch.fft.rfftfreq(size, d=1.0 / size, device=device).to(dtype=dtype)
    return kx[:, None] ** 2 + ky[None, :] ** 2


def _k_2d(size: int, device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
    kx = torch.fft.fftfreq(size, d=1.0 / size, device=device).to(dtype=dtype)
    ky = torch.fft.rfftfreq(size, d=1.0 / size, device=device).to(dtype=dtype)
    return kx[:, None], ky[None, :]


def _expand_param(param: Union[torch.Tensor, float], x: torch.Tensor, spatial_ndim: int) -> torch.Tensor:
    t = torch.as_tensor(param, device=x.device, dtype=x.dtype)
    return t[(...,) + (None,) * spatial_ndim]


def _validate_real_output(x: torch.Tensor, y: torch.Tensor) -> None:
    if y.shape != x.shape:
        raise RuntimeError(f"Output shape mismatch: expected {tuple(x.shape)} got {tuple(y.shape)}")
    if not torch.is_floating_point(y):
        raise RuntimeError("Output must be a real floating tensor")


def apply_heat_semigroup_1d(x: torch.Tensor, tau: Union[torch.Tensor, float], nu: float) -> torch.Tensor:
    if x.ndim < 1:
        raise ValueError("x must have at least 1 dimension")
    size = x.shape[-1]
    k2 = _k_squared_1d(size, x.device, x.dtype)

    tau_t = _expand_param(tau, x, spatial_ndim=1)
    x_ft = torch.fft.rfft(x, dim=-1)
    decay = torch.exp(-nu * tau_t * ((2.0 * math.pi) ** 2) * k2)
    y = torch.fft.irfft(x_ft * decay, n=size, dim=-1)
    _validate_real_output(x, y)
    return y


def apply_advection_1d(x: torch.Tensor, tau: Union[torch.Tensor, float], c: Union[torch.Tensor, float]) -> torch.Tensor:
    if x.ndim < 1:
        raise ValueError("x must have at least 1 dimension")
    size = x.shape[-1]
    k = torch.fft.rfftfreq(size, d=1.0 / size, device=x.device).to(dtype=x.dtype)

    tau_t = _expand_param(tau, x, spatial_ndim=1)
    c_t = _expand_param(c, x, spatial_ndim=1)
    arg = -(2.0 * math.pi) * (tau_t * c_t) * k
    phase = torch.complex(torch.cos(arg), torch.sin(arg))

    x_ft = torch.fft.rfft(x, dim=-1)
    y = torch.fft.irfft(x_ft * phase, n=size, dim=-1)
    _validate_real_output(x, y)
    return y


def apply_convection_diffusion_1d(
    x: torch.Tensor,
    tau: Union[torch.Tensor, float],
    nu: float,
    c: Union[torch.Tensor, float],
) -> torch.Tensor:
    if x.ndim < 1:
        raise ValueError("x must have at least 1 dimension")
    size = x.shape[-1]
    k = torch.fft.rfftfreq(size, d=1.0 / size, device=x.device).to(dtype=x.dtype)
    k2 = k ** 2

    tau_t = _expand_param(tau, x, spatial_ndim=1)
    c_t = _expand_param(c, x, spatial_ndim=1)
    decay = torch.exp(-nu * tau_t * ((2.0 * math.pi) ** 2) * k2)
    arg = -(2.0 * math.pi) * (tau_t * c_t) * k
    phase = torch.complex(torch.cos(arg), torch.sin(arg))

    x_ft = torch.fft.rfft(x, dim=-1)
    y = torch.fft.irfft(x_ft * decay * phase, n=size, dim=-1)
    _validate_real_output(x, y)
    return y


def apply_wave_1d(
    x: torch.Tensor,
    tau: Union[torch.Tensor, float],
    c_wave: Union[torch.Tensor, float],
    gamma: Union[torch.Tensor, float],
) -> torch.Tensor:
    if x.ndim < 1:
        raise ValueError("x must have at least 1 dimension")
    size = x.shape[-1]
    k = torch.fft.rfftfreq(size, d=1.0 / size, device=x.device).to(dtype=x.dtype)

    tau_t = _expand_param(tau, x, spatial_ndim=1)
    c_t = _expand_param(c_wave, x, spatial_ndim=1)
    gamma_t = _expand_param(gamma, x, spatial_ndim=1)
    amp = torch.exp(-0.5 * gamma_t * tau_t)
    osc = torch.cos((2.0 * math.pi) * c_t * tau_t * torch.abs(k))

    x_ft = torch.fft.rfft(x, dim=-1)
    y = torch.fft.irfft(x_ft * (amp * osc), n=size, dim=-1)
    _validate_real_output(x, y)
    return y


def apply_heat_semigroup_2d(x: torch.Tensor, tau: Union[torch.Tensor, float], nu: float) -> torch.Tensor:
    if x.ndim < 2:
        raise ValueError("x must have at least 2 dimensions")
    sx, sy = x.shape[-2], x.shape[-1]
    if sx != sy:
        raise ValueError("Only square grids are supported")
    size = sx
    k2 = _k_squared_2d(size, x.device, x.dtype)

    tau_t = _expand_param(tau, x, spatial_ndim=2)
    x_ft = torch.fft.rfft2(x, dim=(-2, -1))
    decay = torch.exp(-nu * tau_t * ((2.0 * math.pi) ** 2) * k2)
    y = torch.fft.irfft2(x_ft * decay, s=(size, size), dim=(-2, -1))
    _validate_real_output(x, y)
    return y


def apply_helmholtz_2d(x: torch.Tensor, alpha: Union[torch.Tensor, float], nu: float) -> torch.Tensor:
    if x.ndim < 2:
        raise ValueError("x must have at least 2 dimensions")
    sx, sy = x.shape[-2], x.shape[-1]
    if sx != sy:
        raise ValueError("Only square grids are supported")
    size = sx
    k2 = _k_squared_2d(size, x.device, x.dtype)

    alpha_t = _expand_param(alpha, x, spatial_ndim=2)
    denom = alpha_t + nu * ((2.0 * math.pi) ** 2) * k2
    x_ft = torch.fft.rfft2(x, dim=(-2, -1))
    y = torch.fft.irfft2(x_ft / denom, s=(size, size), dim=(-2, -1))
    _validate_real_output(x, y)
    return y


def apply_anisotropic_diffusion_2d(
    x: torch.Tensor,
    tau: Union[torch.Tensor, float],
    d11: Union[torch.Tensor, float],
    d12: Union[torch.Tensor, float],
    d22: Union[torch.Tensor, float],
) -> torch.Tensor:
    if x.ndim < 2:
        raise ValueError("x must have at least 2 dimensions")
    sx, sy = x.shape[-2], x.shape[-1]
    if sx != sy:
        raise ValueError("Only square grids are supported")
    size = sx

    kx, ky = _k_2d(size, x.device, x.dtype)
    tau_t = _expand_param(tau, x, spatial_ndim=2)
    d11_t = _expand_param(d11, x, spatial_ndim=2)
    d12_t = _expand_param(d12, x, spatial_ndim=2)
    d22_t = _expand_param(d22, x, spatial_ndim=2)

    quad = d11_t * (kx ** 2) + 2.0 * d12_t * (kx * ky) + d22_t * (ky ** 2)
    decay = torch.exp(-((2.0 * math.pi) ** 2) * tau_t * quad)

    x_ft = torch.fft.rfft2(x, dim=(-2, -1))
    y = torch.fft.irfft2(x_ft * decay, s=(size, size), dim=(-2, -1))
    _validate_real_output(x, y)
    return y


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
        operator: str = "heat",
        c_dist: str = "uniform",
        c_min: float = -1.0,
        c_max: float = 1.0,
        c_std: float = 1.0,
        c_fixed: float = 1.0,
        wave_c_dist: str = "uniform",
        wave_c_min: float = 0.1,
        wave_c_max: float = 2.0,
        wave_c_fixed: float = 1.0,
        wave_gamma: float = 0.0,
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
        self.operator = operator.lower()

        if self.m <= 0:
            raise ValueError("m must be positive")
        if self.inner_product not in {"mean", "sum"}:
            raise ValueError("inner_product must be one of {'mean', 'sum'}")
        if self.feature_scale not in {"none", "inv_sqrt_m"}:
            raise ValueError("feature_scale must be one of {'none', 'inv_sqrt_m'}")
        if self.operator not in {"heat", "advection", "convdiff", "wave"}:
            raise ValueError("operator must be one of {'heat', 'advection', 'convdiff', 'wave'}")
        if wave_gamma < 0:
            raise ValueError("wave_gamma must be >= 0")

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

        c = torch.zeros(self.m, device=device, dtype=dtype)
        c_wave = torch.zeros(self.m, device=device, dtype=dtype)
        gamma = torch.zeros(self.m, device=device, dtype=dtype)

        if self.operator == "heat":
            h = apply_heat_semigroup_1d(g, tau, self.nu)
        elif self.operator == "advection":
            c = _sample_c(
                self.m,
                dist=c_dist,
                c_min=c_min,
                c_max=c_max,
                c_std=c_std,
                c_fixed=c_fixed,
                device=device,
                dtype=dtype,
            )
            h = apply_advection_1d(g, tau=tau, c=-c)
        elif self.operator == "convdiff":
            c = _sample_c(
                self.m,
                dist=c_dist,
                c_min=c_min,
                c_max=c_max,
                c_std=c_std,
                c_fixed=c_fixed,
                device=device,
                dtype=dtype,
            )
            h = apply_convection_diffusion_1d(g, tau=tau, nu=self.nu, c=-c)
        else:
            c_wave = _sample_alpha(
                self.m,
                dist=wave_c_dist,
                alpha_min=wave_c_min,
                alpha_max=wave_c_max,
                alpha_fixed=wave_c_fixed,
                device=device,
                dtype=dtype,
            )
            gamma = torch.full((self.m,), float(wave_gamma), device=device, dtype=dtype)
            h = apply_wave_1d(g, tau=tau, c_wave=c_wave, gamma=gamma)

        self.register_buffer("tau", tau)
        self.register_buffer("c", c)
        self.register_buffer("c_wave", c_wave)
        self.register_buffer("gamma", gamma)
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
        operator: str = "heat",
        alpha_dist: str = "loguniform",
        alpha_min: float = 1e-2,
        alpha_max: float = 10.0,
        alpha_fixed: float = 1.0,
        aniso_eig_dist: str = "loguniform",
        aniso_eig_min: float = 1e-3,
        aniso_eig_max: float = 1.0,
        aniso_theta_dist: str = "uniform",
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
        self.operator = operator.lower()

        if self.m <= 0:
            raise ValueError("m must be positive")
        if self.inner_product not in {"mean", "sum"}:
            raise ValueError("inner_product must be one of {'mean', 'sum'}")
        if self.feature_scale not in {"none", "inv_sqrt_m"}:
            raise ValueError("feature_scale must be one of {'none', 'inv_sqrt_m'}")
        if self.operator not in {"heat", "helmholtz", "aniso"}:
            raise ValueError("operator must be one of {'heat', 'helmholtz', 'aniso'}")

        if device is None:
            device = torch.device("cpu")

        g = torch.randn(self.m, self.size, self.size, device=device, dtype=dtype)
        if g_smooth_tau > 0:
            g = apply_heat_semigroup_2d(g, float(g_smooth_tau), self.nu)

        tau = torch.zeros(self.m, device=device, dtype=dtype)
        alpha = torch.zeros(self.m, device=device, dtype=dtype)
        d11 = torch.zeros(self.m, device=device, dtype=dtype)
        d12 = torch.zeros(self.m, device=device, dtype=dtype)
        d22 = torch.zeros(self.m, device=device, dtype=dtype)

        if self.operator == "heat":
            tau = _sample_tau(
                self.m,
                tau_dist=tau_dist,
                tau_min=tau_min,
                tau_max=tau_max,
                tau_exp_rate=tau_exp_rate,
                device=device,
                dtype=dtype,
            )
            h = apply_heat_semigroup_2d(g, tau=tau, nu=self.nu)
        elif self.operator == "helmholtz":
            alpha = _sample_alpha(
                self.m,
                dist=alpha_dist,
                alpha_min=alpha_min,
                alpha_max=alpha_max,
                alpha_fixed=alpha_fixed,
                device=device,
                dtype=dtype,
            )
            h = apply_helmholtz_2d(g, alpha=alpha, nu=self.nu)
        else:
            tau = _sample_tau(
                self.m,
                tau_dist=tau_dist,
                tau_min=tau_min,
                tau_max=tau_max,
                tau_exp_rate=tau_exp_rate,
                device=device,
                dtype=dtype,
            )
            d11, d12, d22 = _sample_spd_2x2(
                self.m,
                eig_dist=aniso_eig_dist,
                eig_min=aniso_eig_min,
                eig_max=aniso_eig_max,
                theta_dist=aniso_theta_dist,
                device=device,
                dtype=dtype,
            )
            h = apply_anisotropic_diffusion_2d(g, tau=tau, d11=d11, d12=d12, d22=d22)

        self.register_buffer("tau", tau)
        self.register_buffer("alpha", alpha)
        self.register_buffer("d11", d11)
        self.register_buffer("d12", d12)
        self.register_buffer("d22", d22)
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
