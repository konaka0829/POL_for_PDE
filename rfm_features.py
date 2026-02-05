"""Random feature implementations for RFM baselines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from rfm_poisson import (
    heat_smooth_neumann,
    poisson_solve_dirichlet,
    gradient_centered,
)


def grf_sample_1d(
    m: int,
    k: int,
    *,
    tau: float,
    alpha: float,
    device: torch.device,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Sample 1D periodic Gaussian random fields via spectral filtering."""
    xi = torch.randn((m, k), device=device, generator=generator)
    xi_hat = torch.fft.rfft(xi, norm="ortho")
    freqs = torch.fft.rfftfreq(k, d=1.0 / k, device=device)
    filt = (2.0 * torch.pi * freqs) ** 2 + tau**2
    filt = filt.pow(-alpha / 2.0)
    filt[0] = 0.0
    theta_hat = xi_hat * filt
    theta = torch.fft.irfft(theta_hat, n=k, norm="ortho")
    return theta


def grf_sample_2d(
    m: int,
    s: int,
    *,
    tau: float,
    alpha: float,
    device: torch.device,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Sample 2D periodic Gaussian random fields via spectral filtering."""
    xi = torch.randn((m, s, s), device=device, generator=generator)
    xi_hat = torch.fft.rfft2(xi, norm="ortho")
    kx = torch.fft.fftfreq(s, d=1.0 / s, device=device)
    ky = torch.fft.rfftfreq(s, d=1.0 / s, device=device)
    kx2 = (2.0 * torch.pi * kx) ** 2
    ky2 = (2.0 * torch.pi * ky) ** 2
    filt = (kx2[:, None] + ky2[None, :] + tau**2).pow(-alpha / 2.0)
    filt[0, 0] = 0.0
    theta_hat = xi_hat * filt
    theta = torch.fft.irfft2(theta_hat, s=(s, s), norm="ortho")
    return theta


def burgers_chi_filter(k: int, delta: float, beta: float, device: torch.device) -> torch.Tensor:
    """Compute the Burgers Fourier-space filter chi(k)."""
    ks = torch.arange(0, k // 2 + 1, device=device, dtype=torch.float32)
    r = 2.0 * torch.pi * torch.abs(ks) * delta
    filt = torch.minimum(2.0 * r, (r + 0.5).pow(-beta))
    filt = torch.clamp(filt, min=0.0)
    return filt


@dataclass
class BurgersRFFeatures:
    """Fourier-space random features for Burgers (1D)."""

    theta: torch.Tensor
    delta: float
    beta: float

    def __post_init__(self) -> None:
        k = self.theta.shape[-1]
        device = self.theta.device
        self.theta_hat = torch.fft.rfft(self.theta, norm="ortho")
        self.chi = burgers_chi_filter(k, self.delta, self.beta, device=device)

    def __call__(self, a_batch: torch.Tensor) -> torch.Tensor:
        a_hat = torch.fft.rfft(a_batch, norm="ortho")
        conv_hat = a_hat[:, None, :] * self.theta_hat[None, :, :] * self.chi[None, None, :]
        conv = torch.fft.irfft(conv_hat, n=a_batch.shape[-1], norm="ortho")
        phi = F.elu(conv)
        return phi


def thresholded_sigmoid(
    x: torch.Tensor,
    s_plus: float,
    s_minus: float,
    delta_sig: float,
) -> torch.Tensor:
    return (s_plus - s_minus) / (1.0 + torch.exp(-x / delta_sig)) + s_minus


@dataclass
class DarcyRFFeatures:
    """Predictor-corrector random features for Darcy (2D)."""

    theta1: torch.Tensor
    theta2: torch.Tensor
    s_plus: float
    s_minus: float
    delta_sig: float
    eta: float
    dt: float
    heat_steps: int
    f_const: float
    feature_chunk_size: int = 32

    def __call__(self, a_batch: torch.Tensor) -> torch.Tensor:
        device = a_batch.device
        batch_size, s, _ = a_batch.shape

        theta1 = self.theta1.to(device)
        theta2 = self.theta2.to(device)
        m = theta1.shape[0]

        a_eps = heat_smooth_neumann(a_batch, self.eta, self.dt, self.heat_steps)
        log_a = torch.log(a_eps + 1e-6)
        grad_log_a = gradient_centered(log_a)
        f_over_a = self.f_const / a_batch

        p1 = torch.empty((batch_size, m, s, s), device=device, dtype=a_batch.dtype)
        chunk_size = max(1, min(self.feature_chunk_size, m))
        for j0 in range(0, m, chunk_size):
            j1 = min(j0 + chunk_size, m)
            chunk = j1 - j0
            sigma1_chunk = thresholded_sigmoid(theta1[j0:j1], self.s_plus, self.s_minus, self.delta_sig)
            sigma2_chunk = thresholded_sigmoid(theta2[j0:j1], self.s_plus, self.s_minus, self.delta_sig)

            rhs0_chunk = f_over_a[:, None, :, :] + sigma1_chunk[None, :, :, :]
            rhs0_chunk = rhs0_chunk.reshape(batch_size * chunk, s, s)
            p0_chunk = poisson_solve_dirichlet(rhs0_chunk, s)
            p0_chunk = p0_chunk.reshape(batch_size, chunk, s, s)

            grad_p0_chunk = gradient_centered(p0_chunk.reshape(batch_size * chunk, s, s))
            grad_p0_chunk = (
                grad_p0_chunk[0].reshape(batch_size, chunk, s, s),
                grad_p0_chunk[1].reshape(batch_size, chunk, s, s),
            )

            dot_term = grad_log_a[0][:, None, :, :] * grad_p0_chunk[0] + grad_log_a[1][:, None, :, :] * grad_p0_chunk[1]
            rhs1_chunk = f_over_a[:, None, :, :] + sigma2_chunk[None, :, :, :] + dot_term
            rhs1_chunk = rhs1_chunk.reshape(batch_size * chunk, s, s)
            p1_chunk = poisson_solve_dirichlet(rhs1_chunk, s)
            p1_chunk = p1_chunk.reshape(batch_size, chunk, s, s)
            p1[:, j0:j1] = p1_chunk

        return p1
