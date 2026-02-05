"""rfm_models.py

Random Feature Models (RFMs) for operator learning, matching the methodology in

  * N. H. Nelsen, A. M. Stuart,
    "Operator Learning Using Random Features: A Tool for Scientific Computing"
    (arXiv:2408.06526)

This module is intentionally designed to *fit the style* of POL_for_PDE:
  - Pure PyTorch implementation (CPU/GPU)
  - Data are expected to be discretized on uniform grids
  - Training is convex (quadratic) via (regularized) normal equations

We implement two feature maps from the paper:
  (1) Fourier-space random features for 1D Burgers (eq. 3.5–3.7)
  (2) Predictor-corrector random features for 2D Darcy flow (eq. 3.12–3.14)

The training inner products are taken in L2 and approximated by composite
trapezoidal rule (as in eq. 4.4 of the paper).

Notes
-----
* This file is based on the shared reference repo "error-bounds-for-vvRF" for
  the 1D Burgers part, but reworked to:
    - use a more vectorized Gram accumulation (via quadrature weights)
    - expose a clean API used by rfm_1d.py / rfm_2d.py

* The 2D Darcy part is implemented from the paper description; it relies on
  a fast Poisson solver with Dirichlet boundary conditions using a DST-I
  implementation built from FFTs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# Quadrature helpers (composite trapezoid rule)
# -----------------------------------------------------------------------------


def trapezoid_weights_1d(K: int, *, h: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """1D trapezoid weights for uniform grid with spacing h.

    Matches torch.trapz(y, dx=h) for a tensor of length K.
    """
    w = torch.full((K,), float(h), device=device, dtype=dtype)
    if K >= 2:
        w[0] *= 0.5
        w[-1] *= 0.5
    return w


def trapezoid_weights_2d(N: int, *, h: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """2D trapezoid weights on an N x N grid with spacing h in each direction.

    Returns W with shape (N, N) where
      \int f(x,y) dx dy  ~  sum_{i,j} W[i,j] f[i,j].

    We use the tensor-product trapezoid rule.
    """
    w1 = trapezoid_weights_1d(N, h=h, device=device, dtype=dtype)
    return torch.einsum("i,j->ij", w1, w1)


def weighted_l2_norm(x: torch.Tensor, w: torch.Tensor, *, eps: float = 1e-12) -> torch.Tensor:
    """Weighted L2 norm over the last dimensions matching w.

    x: (..., K) with w:(K,)  OR  x:(..., N, N) with w:(N,N)
    returns: (...) norms
    """
    w_ndim = w.ndim
    while w.ndim < x.ndim:
        w = w.unsqueeze(0)
    return torch.sqrt(torch.sum(w * (x**2), dim=tuple(range(-w_ndim, 0))) + eps)


def relative_l2_error(pred: torch.Tensor, gt: torch.Tensor, w: torch.Tensor, *, eps: float = 1e-12) -> torch.Tensor:
    """Per-sample relative L2 error with trapezoid weights."""
    num = weighted_l2_norm(pred - gt, w, eps=eps)
    den = weighted_l2_norm(gt, w, eps=eps)
    return num / (den + eps)


# -----------------------------------------------------------------------------
# 1D: Burgers — Fourier-space random features (eq. 3.5–3.7)
# -----------------------------------------------------------------------------


class GaussianRFcoeff1D:
    """Periodic GRF coefficients for real-valued fields using rfft convention.

    This matches the reference implementation.
    """

    def __init__(self, size: int, *, kmax: Optional[int] = None, device: Optional[torch.device] = None):
        if size % 2 != 0:
            raise ValueError("size must be even (power-of-two recommended).")
        self.device = device
        self.size = size
        self.kfreq = size // 2
        if kmax is None:
            self.kmax = self.kfreq
        else:
            self.kmax = min(int(kmax), self.kfreq)

    def sample(self, n: int) -> torch.Tensor:
        """Return complex coefficients for positive frequencies (excluding k=0).

        Output shape: (n, kmax)
        """
        iid = torch.randn(n, self.kmax, 2, device=self.device)
        return (iid[..., 0] - iid[..., 1] * 1.0j) / 2.0

    def zeropad(self, ctensor: torch.Tensor) -> torch.Tensor:
        """Pad coefficients (..., kmax) into rfft layout (..., kfreq+1)."""
        coeff = torch.zeros(*ctensor.shape[:-1], self.kfreq + 1, dtype=torch.cfloat, device=self.device)
        coeff[..., 1 : self.kmax + 1] = ctensor
        return coeff


@dataclass
class RFM1DConfig:
    K: int
    K_fine: int
    m: int
    lamreg: float = 0.0
    # RF filter hyperparameters (paper: delta=0.0025, beta=4)
    nu_rf: float = 2.5e-3
    al_rf: float = 4.0
    sig_rf: float = 1.0
    # GRF hyperparameters for theta ~ N(0,C') (paper: tau'=5, alpha'=2)
    tau_g: float = 5.0
    al_g: float = 2.0
    sig_g: Optional[float] = None
    kmax: Optional[int] = None
    # batching
    batch_size: int = 20
    feature_batch_size: int = 64
    feature_batch_size_test: int = 128
    device: torch.device = torch.device("cpu")
    dtype: torch.dtype = torch.float32


class RandomFeatureModel1D(torch.nn.Module):
    """Vector-valued RFM for 1D Burgers operator.

    Implements feature map (3.5) with ELU activation and filter (3.6).
    """

    def __init__(self, cfg: RFM1DConfig):
        super().__init__()
        self.cfg = cfg

        if cfg.K % 2 != 0:
            raise ValueError("RFM1D requires even K (FFT-friendly).")
        self.K = int(cfg.K)
        self.K_fine = int(cfg.K_fine)
        self.m = int(cfg.m)
        self.lamreg = float(cfg.lamreg)
        self.nu_rf = float(cfg.nu_rf)
        self.al_rf = float(cfg.al_rf)
        self.sig_rf = float(cfg.sig_rf)
        self.tau_g = float(cfg.tau_g)
        self.al_g = float(cfg.al_g)
        self.sig_g = (
            float(cfg.sig_g)
            if cfg.sig_g is not None
            else float(cfg.tau_g ** (0.5 * (2.0 * cfg.al_g - 1.0)))
        )
        self.device = cfg.device
        self.dtype = cfg.dtype

        self.kmax = int(cfg.kmax) if cfg.kmax is not None else self.K // 2
        self.kmax = min(self.kmax, self.K // 2)

        # quadrature weights: paper uses composite trapezoid; reference uses h=1/K
        self.h = 1.0 / float(self.K)
        w = trapezoid_weights_1d(self.K, h=self.h, device=self.device, dtype=self.dtype)
        self.register_buffer("w", w)
        self.register_buffer("sqrt_w", torch.sqrt(w))

        # wavenumbers for rfft
        kwave = torch.arange(0, self.K // 2 + 1, device=self.device, dtype=self.dtype)
        self.register_buffer("kwave", kwave)

        # random Fourier coefficients for theta
        self.grf = GaussianRFcoeff1D(self.K, kmax=self.kmax, device=self.device)
        grf_g = torch.zeros((self.m, self.kmax), dtype=torch.cfloat, device=self.device)
        self.register_buffer("grf_g", grf_g)
        self.resample()

        # learned coefficients alpha (length m)
        self.register_buffer("alpha", torch.zeros(self.m, device=self.device, dtype=self.dtype))

    # ------------------------------------------------------------------
    # Random feature map components
    # ------------------------------------------------------------------

    @staticmethod
    def _act_filter(r: torch.Tensor, al_rf: float) -> torch.Tensor:
        """Filter activation σχ from eq. (3.6)."""
        return F.relu(torch.minimum(2 * r, torch.pow(0.5 + r, -al_rf)))

    def resample(self) -> None:
        """Resample the iid random Fourier coefficients for theta."""
        feat_bs = max(1, int(self.cfg.feature_batch_size_test))
        for sl in torch.split(self.grf_g, feat_bs, dim=0):
            sl[...] = self.grf.sample(sl.shape[0])

    def rf_batch(self, a_batch: torch.Tensor, g_batch: torch.Tensor) -> torch.Tensor:
        """Evaluate Fourier-space random features for a batch.

        Inputs:
            a_batch: (nbatch, K) real
            g_batch: (mbatch, kmax) complex (positive freq coeffs)
        Returns:
            (nbatch, mbatch, K) real
        """
        PI = math.pi
        sqrt_eig = self.sig_g * (
            (4.0 * (PI**2) * (self.kwave[1 : self.kmax + 1] ** 2) + self.tau_g**2) ** (-self.al_g / 2.0)
        )

        wave_func = self._act_filter(torch.abs(self.nu_rf * self.kwave * 2.0 * PI), self.al_rf)

        a_ft = torch.fft.rfft(a_batch)

        g_scaled = math.sqrt(2.0) * sqrt_eig * g_batch

        conv = torch.einsum("nk,mk->nmk", a_ft[..., 1 : self.kmax + 1], g_scaled)

        conv_pad = self.grf.zeropad(conv)

        out = torch.fft.irfft(wave_func * conv_pad, n=self.K)

        return self.sig_rf * F.elu(self.K_fine * out)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, x_train: torch.Tensor, y_train: torch.Tensor) -> None:
        """Train alpha by solving the (regularized) normal equations."""

        from torch.utils.data import DataLoader, TensorDataset

        x_train = x_train.to(self.device, dtype=self.dtype)
        y_train = y_train.to(self.device, dtype=self.dtype)

        loader = DataLoader(
            TensorDataset(x_train, y_train),
            batch_size=min(int(self.cfg.batch_size), x_train.shape[0]),
            shuffle=True,
        )

        AstarY = torch.zeros(self.m, device=self.device, dtype=self.dtype)
        AstarA = torch.zeros((self.m, self.m), device=self.device, dtype=self.dtype)

        sqrt_w = self.sqrt_w

        feat_bs = max(1, int(self.cfg.feature_batch_size))

        for a, y in loader:
            a = a.to(self.device, dtype=self.dtype)
            y = y.to(self.device, dtype=self.dtype)

            y_w = y * sqrt_w

            stored: List[Tuple[int, int, torch.Tensor]] = []

            for j0 in range(0, self.m, feat_bs):
                j1 = min(self.m, j0 + feat_bs)
                g = self.grf_g[j0:j1]
                RF = self.rf_batch(a, g)
                Phi = RF * sqrt_w

                AstarY[j0:j1] += torch.einsum("nmk,nk->m", Phi, y_w)

                for i0, i1, Phi_prev in stored:
                    AstarA[i0:i1, j0:j1] += torch.einsum("nmk,njk->mj", Phi_prev, Phi)

                AstarA[j0:j1, j0:j1] += torch.einsum("nmk,njk->mj", Phi, Phi)

                stored.append((j0, j1, Phi))

        AstarA = AstarA + AstarA.T - torch.diag(torch.diag(AstarA))
        AstarA = AstarA / float(self.m)

        if self.lamreg > 0.0:
            A_reg = AstarA + self.lamreg * torch.eye(self.m, device=self.device, dtype=self.dtype)
            alpha = torch.linalg.solve(A_reg, AstarY)
        else:
            alpha = torch.linalg.pinv(AstarA) @ AstarY

        self.alpha[...] = alpha

    # ------------------------------------------------------------------
    # Prediction + metrics
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict(self, a: torch.Tensor, *, feature_batch_size: Optional[int] = None) -> torch.Tensor:
        """Predict on a batch of inputs.

        a: (nbatch, K) or (K,)
        returns: (nbatch, K) or (K,)
        """
        single = False
        if a.ndim == 1:
            single = True
            a = a.unsqueeze(0)
        a = a.to(self.device, dtype=self.dtype)

        nbatch = a.shape[0]
        out = torch.zeros((nbatch, self.K), device=self.device, dtype=self.dtype)

        feat_bs = int(feature_batch_size or self.cfg.feature_batch_size_test)
        feat_bs = max(1, feat_bs)

        for j0 in range(0, self.m, feat_bs):
            j1 = min(self.m, j0 + feat_bs)
            g = self.grf_g[j0:j1]
            RF = self.rf_batch(a, g)
            al = self.alpha[j0:j1]
            out += torch.einsum("m,nmk->nk", al, RF)

        out = out / float(self.m)
        return out.squeeze(0) if single else out

    @torch.no_grad()
    def per_sample_relative_errors(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute per-sample relative L2 error on a dataset."""
        x = x.to(self.device, dtype=self.dtype)
        y = y.to(self.device, dtype=self.dtype)
        pred = self.predict(x)
        return relative_l2_error(pred, y, self.w)


# -----------------------------------------------------------------------------
# 2D: Darcy — predictor-corrector random features (eq. 3.12–3.14)
# -----------------------------------------------------------------------------


def _dst_type1(x: torch.Tensor) -> torch.Tensor:
    """DST-I along the last dimension, implemented via FFT.

    This implementation follows the standard odd-extension trick.
    For a vector x of length n, DST-I is defined (unnormalized) as
        X_k = \sum_{j=1}^n x_j sin(pi*k*j/(n+1)),  k=1..n
    (with 1-based indexing).
    """
    n = x.shape[-1]
    y = torch.zeros(*x.shape[:-1], 2 * (n + 1), device=x.device, dtype=x.dtype)
    y[..., 1 : n + 1] = x
    y[..., n + 2 :] = -torch.flip(x, dims=[-1])
    Y = torch.fft.fft(y, dim=-1)
    return -Y[..., 1 : n + 1].imag


def _idst_type1(X: torch.Tensor) -> torch.Tensor:
    """Inverse of DST-I (up to the standard scaling)."""
    n = X.shape[-1]
    return _dst_type1(X) / (2.0 * (n + 1))


def _dst2_type1(x: torch.Tensor) -> torch.Tensor:
    """2D DST-I on the last two dims."""
    y = _dst_type1(x)
    y = _dst_type1(y.transpose(-1, -2)).transpose(-1, -2)
    return y


def _idst2_type1(X: torch.Tensor) -> torch.Tensor:
    """Inverse 2D DST-I on the last two dims."""
    y = _idst_type1(X)
    y = _idst_type1(y.transpose(-1, -2)).transpose(-1, -2)
    return y


class PoissonSolver2DDirichlet:
    """Fast Poisson solver for -Δu = f on (0,1)^2 with homogeneous Dirichlet BC.

    Uses a second-order finite-difference stencil + DST-I diagonalization.
    """

    def __init__(self, N: int, *, device: torch.device, dtype: torch.dtype):
        if N < 3:
            raise ValueError("Need N>=3 to have interior points.")
        self.N = int(N)
        self.n = self.N - 2
        self.device = device
        self.dtype = dtype
        self.h = 1.0 / float(self.N - 1)

        k = torch.arange(1, self.n + 1, device=device, dtype=dtype)
        lam_1d = 2.0 - 2.0 * torch.cos(math.pi * k / float(self.n + 1))
        denom = (lam_1d[:, None] + lam_1d[None, :]) / (self.h**2)
        self.register = {}
        self.register["denom"] = denom

    @property
    def denom(self) -> torch.Tensor:
        return self.register["denom"]

    def solve(self, rhs: torch.Tensor) -> torch.Tensor:
        """Solve -Δu = rhs for u with u=0 on boundary.

        rhs: (..., N, N)
        returns u: same shape, with boundary zeros.
        """
        rhs = rhs.to(self.device, dtype=self.dtype)
        rhs_int = rhs[..., 1:-1, 1:-1]
        rhs_hat = _dst2_type1(rhs_int)
        u_hat = rhs_hat / self.denom
        u_int = _idst2_type1(u_hat)
        u = torch.zeros_like(rhs)
        u[..., 1:-1, 1:-1] = u_int
        return u


def laplacian_neumann_2d(u: torch.Tensor, *, h: float) -> torch.Tensor:
    """2D Laplacian with homogeneous Neumann BC via replicate padding."""
    N = u.shape[-1]
    if u.shape[-2] != N:
        raise ValueError("u must be (..., N, N)")
    u_flat = u.reshape(-1, 1, N, N)
    u_pad = F.pad(u_flat, (1, 1, 1, 1), mode="replicate")
    lap = (
        u_pad[:, :, 2:, 1:-1]
        + u_pad[:, :, :-2, 1:-1]
        + u_pad[:, :, 1:-1, 2:]
        + u_pad[:, :, 1:-1, :-2]
        - 4.0 * u_flat
    ) / (h**2)
    return lap.reshape(u.shape)


def gradient_central_2d(u: torch.Tensor, *, h: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """Centered finite-difference gradient on a uniform grid.

    Returns (du/dx, du/dy) with the same shape as u.
    """
    dudx = torch.zeros_like(u)
    dudy = torch.zeros_like(u)
    dudx[..., 1:-1, :] = (u[..., 2:, :] - u[..., :-2, :]) / (2.0 * h)
    dudx[..., 0, :] = (u[..., 1, :] - u[..., 0, :]) / h
    dudx[..., -1, :] = (u[..., -1, :] - u[..., -2, :]) / h
    dudy[..., :, 1:-1] = (u[..., :, 2:] - u[..., :, :-2]) / (2.0 * h)
    dudy[..., :, 0] = (u[..., :, 1] - u[..., :, 0]) / h
    dudy[..., :, -1] = (u[..., :, -1] - u[..., :, -2]) / h
    return dudx, dudy


@dataclass
class RFM2DConfig:
    N: int
    m: int
    lamreg: float = 1e-8
    tau_g: float = 7.5
    al_g: float = 2.0
    sig_g: Optional[float] = None
    s_plus: float = 1.0 / 12.0
    s_minus: float = -1.0 / 3.0
    delta_sig: float = 0.15
    smooth_dt: float = 0.03
    smooth_eta: float = 1e-4
    smooth_steps: int = 34
    batch_size: int = 10
    feature_batch_size: int = 8
    feature_batch_size_test: int = 16
    device: torch.device = torch.device("cpu")
    dtype: torch.dtype = torch.float32
    eps: float = 1e-9
    f_value: float = 1.0


class PredictorCorrectorRFM2D(torch.nn.Module):
    """Vector-valued RFM for 2D Darcy operator using predictor-corrector features."""

    def __init__(self, cfg: RFM2DConfig):
        super().__init__()
        self.cfg = cfg
        self.N = int(cfg.N)
        self.m = int(cfg.m)
        self.lamreg = float(cfg.lamreg)
        self.tau_g = float(cfg.tau_g)
        self.al_g = float(cfg.al_g)
        d = 2.0
        self.sig_g = (
            float(cfg.sig_g)
            if cfg.sig_g is not None
            else float(cfg.tau_g ** (0.5 * (2.0 * cfg.al_g - d)))
        )
        self.s_plus = float(cfg.s_plus)
        self.s_minus = float(cfg.s_minus)
        self.delta_sig = float(cfg.delta_sig)
        self.device = cfg.device
        self.dtype = cfg.dtype
        self.eps = float(cfg.eps)
        self.f_value = float(cfg.f_value)

        self.h = 1.0 / float(self.N - 1)
        W = trapezoid_weights_2d(self.N, h=self.h, device=self.device, dtype=self.dtype)
        self.register_buffer("W", W)
        self.register_buffer("sqrt_W", torch.sqrt(W))

        self.register_buffer("_grf_mult", self._build_grf_multiplier())

        self.poisson = PoissonSolver2DDirichlet(self.N, device=self.device, dtype=self.dtype)

        self.register_buffer("theta1", torch.zeros((self.m, self.N, self.N), device=self.device, dtype=self.dtype))
        self.register_buffer("theta2", torch.zeros((self.m, self.N, self.N), device=self.device, dtype=self.dtype))
        self.resample()

        self.register_buffer("alpha", torch.zeros(self.m, device=self.device, dtype=self.dtype))

    # ------------------------------------------------------------------
    # GRF sampling (periodic via rfft2)
    # ------------------------------------------------------------------

    def _build_grf_multiplier(self) -> torch.Tensor:
        """Spectral multiplier sqrt(eig) for periodic Matérn-like GRF."""
        kx = torch.fft.fftfreq(self.N, d=1.0 / self.N).to(self.device, dtype=self.dtype)
        ky = torch.fft.rfftfreq(self.N, d=1.0 / self.N).to(self.device, dtype=self.dtype)
        kx2 = kx[:, None] ** 2
        ky2 = ky[None, :] ** 2
        PI = math.pi
        lap = 4.0 * (PI**2) * (kx2 + ky2)
        mult = self.sig_g * (lap + self.tau_g**2) ** (-self.al_g / 2.0)
        mult[0, 0] = 0.0
        return mult

    def _sample_grf(self, n: int) -> torch.Tensor:
        """Sample n GRFs with the configured covariance on an N x N grid."""
        z = torch.randn((n, self.N, self.N), device=self.device, dtype=self.dtype)
        z_hat = torch.fft.rfft2(z, norm="forward")
        u_hat = z_hat * self._grf_mult
        u = torch.fft.irfft2(u_hat, s=(self.N, self.N), norm="forward")
        return u

    def resample(self) -> None:
        """Resample theta1/theta2."""
        feat_bs = max(1, int(self.cfg.feature_batch_size_test))
        for j0 in range(0, self.m, feat_bs):
            j1 = min(self.m, j0 + feat_bs)
            self.theta1[j0:j1] = self._sample_grf(j1 - j0)
            self.theta2[j0:j1] = self._sample_grf(j1 - j0)

    # ------------------------------------------------------------------
    # Feature map helpers
    # ------------------------------------------------------------------

    def sigma_gamma(self, r: torch.Tensor) -> torch.Tensor:
        """Thresholded sigmoid σ_γ from eq. (3.13)."""
        return (self.s_plus - self.s_minus) / (1.0 + torch.exp(-r / self.delta_sig)) + self.s_minus

    def smooth_coefficient(self, a: torch.Tensor) -> torch.Tensor:
        """Smooth a via the heat equation (3.14) using explicit FD steps."""
        v = a.to(self.device, dtype=self.dtype)
        dt = float(self.cfg.smooth_dt)
        eta = float(self.cfg.smooth_eta)
        steps = int(self.cfg.smooth_steps)
        for _ in range(steps):
            v = v + dt * eta * laplacian_neumann_2d(v, h=self.h)
        return v

    def _prepare_a(self, a: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Compute a_eps and grad(log a_eps) for a batch."""
        a_eps = self.smooth_coefficient(a)
        loga = torch.log(a_eps.clamp_min(self.eps))
        dlogx, dlogy = gradient_central_2d(loga, h=self.h)
        return a_eps, (dlogx, dlogy)

    def rf_batch(
        self,
        a_eps: torch.Tensor,
        grad_log_a: Tuple[torch.Tensor, torch.Tensor],
        theta1: torch.Tensor,
        theta2: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate predictor-corrector random features.

        a_eps: (nbatch, N, N)
        grad_log_a: (dlogx, dlogy) each (nbatch, N, N)
        theta1/theta2: (mchunk, N, N)
        returns: (nbatch, mchunk, N, N)
        """
        dlogx, dlogy = grad_log_a

        sig1 = self.sigma_gamma(theta1)
        sig2 = self.sigma_gamma(theta2)

        denom0 = a_eps[:, None, :, :] + sig1[None, :, :, :]
        rhs0 = self.f_value / denom0
        p0 = self.poisson.solve(rhs0)

        dp0x, dp0y = gradient_central_2d(p0, h=self.h)
        corr = dlogx[:, None, :, :] * dp0x + dlogy[:, None, :, :] * dp0y

        denom1 = a_eps[:, None, :, :] + sig2[None, :, :, :]
        rhs1 = self.f_value / denom1 + corr
        p1 = self.poisson.solve(rhs1)

        return p1

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, x_train: torch.Tensor, y_train: torch.Tensor) -> None:
        """Train alpha by solving the (regularized) normal equations."""
        from torch.utils.data import DataLoader, TensorDataset

        x_train = x_train.to(self.device, dtype=self.dtype)
        y_train = y_train.to(self.device, dtype=self.dtype)

        loader = DataLoader(
            TensorDataset(x_train, y_train),
            batch_size=min(int(self.cfg.batch_size), x_train.shape[0]),
            shuffle=True,
        )

        AstarY = torch.zeros(self.m, device=self.device, dtype=self.dtype)
        AstarA = torch.zeros((self.m, self.m), device=self.device, dtype=self.dtype)

        sqrt_W = self.sqrt_W
        feat_bs = max(1, int(self.cfg.feature_batch_size))

        for a, y in loader:
            a = a.to(self.device, dtype=self.dtype)
            y = y.to(self.device, dtype=self.dtype)

            a_eps, grad_log_a = self._prepare_a(a)
            y_w = y * sqrt_W

            stored: List[Tuple[int, int, torch.Tensor]] = []

            for j0 in range(0, self.m, feat_bs):
                j1 = min(self.m, j0 + feat_bs)
                th1 = self.theta1[j0:j1]
                th2 = self.theta2[j0:j1]
                RF = self.rf_batch(a_eps, grad_log_a, th1, th2)
                Phi = RF * sqrt_W

                AstarY[j0:j1] += torch.einsum("nmxy,nxy->m", Phi, y_w)

                for i0, i1, Phi_prev in stored:
                    AstarA[i0:i1, j0:j1] += torch.einsum("nmxy,njxy->mj", Phi_prev, Phi)

                AstarA[j0:j1, j0:j1] += torch.einsum("nmxy,njxy->mj", Phi, Phi)

                stored.append((j0, j1, Phi))

        AstarA = AstarA + AstarA.T - torch.diag(torch.diag(AstarA))
        AstarA = AstarA / float(self.m)

        if self.lamreg > 0.0:
            A_reg = AstarA + self.lamreg * torch.eye(self.m, device=self.device, dtype=self.dtype)
            alpha = torch.linalg.solve(A_reg, AstarY)
        else:
            alpha = torch.linalg.pinv(AstarA) @ AstarY

        self.alpha[...] = alpha

    # ------------------------------------------------------------------
    # Prediction + metrics
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict(self, a: torch.Tensor, *, feature_batch_size: Optional[int] = None) -> torch.Tensor:
        """Predict u for a batch of coefficients a.

        a: (nbatch, N, N) or (N, N)
        returns: same shape
        """
        single = False
        if a.ndim == 2:
            single = True
            a = a.unsqueeze(0)
        a = a.to(self.device, dtype=self.dtype)

        a_eps, grad_log_a = self._prepare_a(a)
        nbatch = a.shape[0]
        out = torch.zeros((nbatch, self.N, self.N), device=self.device, dtype=self.dtype)

        feat_bs = int(feature_batch_size or self.cfg.feature_batch_size_test)
        feat_bs = max(1, feat_bs)

        for j0 in range(0, self.m, feat_bs):
            j1 = min(self.m, j0 + feat_bs)
            th1 = self.theta1[j0:j1]
            th2 = self.theta2[j0:j1]
            RF = self.rf_batch(a_eps, grad_log_a, th1, th2)
            al = self.alpha[j0:j1]
            out += torch.einsum("m,nmxy->nxy", al, RF)

        out = out / float(self.m)
        return out.squeeze(0) if single else out

    @torch.no_grad()
    def per_sample_relative_errors(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device, dtype=self.dtype)
        y = y.to(self.device, dtype=self.dtype)
        pred = self.predict(x)
        return relative_l2_error(pred, y, self.W)
