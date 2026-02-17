"""Utilities for backprop-free Koopman-Reservoir operator learning (1D/2D).

This module provides:
  - basis construction for measurement/decoder
  - fixed random reservoir encoder
  - ridge solvers for Koopman/decoder
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch


def set_global_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("--device=cuda was requested, but CUDA is not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def estimate_spectral_radius(mat: torch.Tensor) -> float:
    eigvals = torch.linalg.eigvals(mat)
    return float(torch.max(torch.abs(eigvals)).real.item())


def normalize_basis_rows(phi: torch.Tensor, dx: float, eps: float = 1e-12) -> torch.Tensor:
    # phi: [Q, S]
    norms = torch.sqrt(torch.sum(phi * phi, dim=1, keepdim=True) * dx + eps)
    return phi / norms


def build_basis_1d(
    basis_name: str,
    dim: int,
    x_grid: torch.Tensor,
    *,
    normalize: bool = False,
    rbf_sigma: float = 0.05,
    random_fourier_scale: float = 4.0,
    seed: int = 0,
) -> torch.Tensor:
    # x_grid: [S] in [0, 1]
    if dim <= 0:
        raise ValueError(f"basis dim must be positive, got {dim}.")
    if x_grid.ndim != 1:
        raise ValueError(f"x_grid must be 1D, got shape={tuple(x_grid.shape)}")

    device = x_grid.device
    dtype = x_grid.dtype
    S = x_grid.shape[0]
    dx = float(1.0 / max(S - 1, 1))
    x = x_grid

    if basis_name == "fourier":
        phi = torch.zeros((dim, S), dtype=dtype, device=device)
        phi[0] = 1.0
        for j in range(1, dim):
            k = (j + 1) // 2
            ang = 2.0 * np.pi * k * x
            phi[j] = torch.cos(ang) if (j % 2 == 1) else torch.sin(ang)
    elif basis_name == "random_fourier":
        rng = np.random.default_rng(seed)
        omega = torch.tensor(
            rng.normal(loc=0.0, scale=random_fourier_scale, size=(dim, 1)),
            dtype=dtype,
            device=device,
        )
        phase = torch.tensor(
            rng.uniform(low=0.0, high=2.0 * np.pi, size=(dim, 1)),
            dtype=dtype,
            device=device,
        )
        phi = np.sqrt(2.0) * torch.cos(2.0 * np.pi * omega * x[None, :] + phase)
    elif basis_name == "legendre":
        t = 2.0 * x - 1.0  # [0,1] -> [-1,1]
        phi = torch.zeros((dim, S), dtype=dtype, device=device)
        phi[0] = 1.0
        if dim > 1:
            phi[1] = t
        for n in range(2, dim):
            # P_n = ((2n-1)tP_{n-1} - (n-1)P_{n-2}) / n
            phi[n] = ((2 * n - 1) * t * phi[n - 1] - (n - 1) * phi[n - 2]) / n
    elif basis_name == "chebyshev":
        t = 2.0 * x - 1.0  # [0,1] -> [-1,1]
        phi = torch.zeros((dim, S), dtype=dtype, device=device)
        phi[0] = 1.0
        if dim > 1:
            phi[1] = t
        for n in range(2, dim):
            # T_n = 2 t T_{n-1} - T_{n-2}
            phi[n] = 2.0 * t * phi[n - 1] - phi[n - 2]
    elif basis_name == "rbf":
        centers = torch.linspace(0.0, 1.0, dim, dtype=dtype, device=device)
        if rbf_sigma <= 0:
            raise ValueError(f"--rbf-sigma must be positive, got {rbf_sigma}.")
        diff = x[None, :] - centers[:, None]
        phi = torch.exp(-(diff * diff) / (2.0 * (rbf_sigma**2)))
    elif basis_name == "sensor":
        # One-hot basis scaled by 1/dx so that <u, psi_j> approximates point samples.
        idx = torch.linspace(0, S - 1, dim, device=device).round().long()
        phi = torch.zeros((dim, S), dtype=dtype, device=device)
        phi[torch.arange(dim, device=device), idx] = 1.0 / max(dx, 1e-12)
    else:
        raise ValueError(f"Unknown basis '{basis_name}'.")

    if normalize:
        phi = normalize_basis_rows(phi, dx=dx)
    return phi


def _build_poly_basis_1d(kind: str, dim: int, grid_1d: torch.Tensor) -> torch.Tensor:
    if dim <= 0:
        raise ValueError(f"basis dim must be positive, got {dim}.")
    if grid_1d.ndim != 1:
        raise ValueError(f"grid must be 1D, got shape={tuple(grid_1d.shape)}")

    t = 2.0 * grid_1d - 1.0  # [0,1] -> [-1,1]
    phi = torch.zeros((dim, grid_1d.shape[0]), dtype=grid_1d.dtype, device=grid_1d.device)
    phi[0] = 1.0
    if dim > 1:
        phi[1] = t

    if kind == "legendre":
        for n in range(2, dim):
            phi[n] = ((2 * n - 1) * t * phi[n - 1] - (n - 1) * phi[n - 2]) / n
        return phi
    if kind == "chebyshev":
        for n in range(2, dim):
            phi[n] = 2.0 * t * phi[n - 1] - phi[n - 2]
        return phi
    raise ValueError(f"Unsupported polynomial basis '{kind}'.")


def build_basis_2d(
    basis_name: str,
    dim: int,
    x_grid: torch.Tensor,
    y_grid: torch.Tensor,
    *,
    normalize: bool = False,
    rbf_sigma: float = 0.05,
    random_fourier_scale: float = 4.0,
    seed: int = 0,
) -> torch.Tensor:
    # return phi: [dim, S] where S = sx * sy
    if dim <= 0:
        raise ValueError(f"basis dim must be positive, got {dim}.")
    if x_grid.ndim != 1 or y_grid.ndim != 1:
        raise ValueError(
            f"x_grid/y_grid must be 1D, got {tuple(x_grid.shape)} and {tuple(y_grid.shape)}"
        )
    if rbf_sigma <= 0 and basis_name == "rbf":
        raise ValueError(f"--rbf-sigma must be positive, got {rbf_sigma}.")

    device = x_grid.device
    dtype = x_grid.dtype
    sx = int(x_grid.shape[0])
    sy = int(y_grid.shape[0])
    dx = float(1.0 / max(sx - 1, 1))
    dy = float(1.0 / max(sy - 1, 1))
    dxdy = dx * dy

    xx, yy = torch.meshgrid(x_grid, y_grid, indexing="ij")
    x_flat = xx.reshape(-1)
    y_flat = yy.reshape(-1)
    s_total = x_flat.shape[0]

    if basis_name == "fourier":
        rows: list[torch.Tensor] = [torch.ones((s_total,), dtype=dtype, device=device)]
        seen_wavevectors: set[tuple[int, int]] = set()
        k_max = 0
        while len(rows) < dim:
            k_max += 1
            wavevectors: list[tuple[int, int]] = []
            for kx in range(-k_max, k_max + 1):
                for ky in range(-k_max, k_max + 1):
                    if kx == 0 and ky == 0:
                        continue
                    # Canonical half-space to avoid +/- duplicates.
                    if (kx > 0 or (kx == 0 and ky > 0)) and (kx, ky) not in seen_wavevectors:
                        seen_wavevectors.add((kx, ky))
                        wavevectors.append((kx, ky))
            wavevectors.sort(key=lambda kk: (abs(kk[0]) + abs(kk[1]), max(abs(kk[0]), abs(kk[1]))))
            for kx, ky in wavevectors:
                if len(rows) >= dim:
                    break
                ang = 2.0 * np.pi * (float(kx) * x_flat + float(ky) * y_flat)
                rows.append(torch.cos(ang))
                if len(rows) >= dim:
                    break
                rows.append(torch.sin(ang))
        phi = torch.stack(rows[:dim], dim=0)
    elif basis_name == "random_fourier":
        rng = np.random.default_rng(seed)
        omega = torch.tensor(
            rng.normal(loc=0.0, scale=random_fourier_scale, size=(dim, 2)),
            dtype=dtype,
            device=device,
        )
        phase = torch.tensor(
            rng.uniform(low=0.0, high=2.0 * np.pi, size=(dim, 1)),
            dtype=dtype,
            device=device,
        )
        arg = 2.0 * np.pi * (omega[:, 0:1] * x_flat[None, :] + omega[:, 1:2] * y_flat[None, :]) + phase
        phi = np.sqrt(2.0) * torch.cos(arg)
    elif basis_name in {"legendre", "chebyshev"}:
        n_side = int(np.ceil(np.sqrt(dim)))
        bx = _build_poly_basis_1d(basis_name, n_side, x_grid)  # [n_side, sx]
        by = _build_poly_basis_1d(basis_name, n_side, y_grid)  # [n_side, sy]
        rows = []
        for i in range(n_side):
            for j in range(n_side):
                if len(rows) >= dim:
                    break
                rows.append((bx[i][:, None] * by[j][None, :]).reshape(-1))
            if len(rows) >= dim:
                break
        phi = torch.stack(rows, dim=0)
    elif basis_name == "rbf":
        n_side = int(np.ceil(np.sqrt(dim)))
        cx = torch.linspace(0.0, 1.0, n_side, dtype=dtype, device=device)
        cy = torch.linspace(0.0, 1.0, n_side, dtype=dtype, device=device)
        cxx, cyy = torch.meshgrid(cx, cy, indexing="ij")
        centers = torch.stack([cxx.reshape(-1), cyy.reshape(-1)], dim=1)[:dim]  # [dim,2]
        diff_x = x_flat[None, :] - centers[:, 0:1]
        diff_y = y_flat[None, :] - centers[:, 1:2]
        phi = torch.exp(-(diff_x * diff_x + diff_y * diff_y) / (2.0 * (rbf_sigma**2)))
    elif basis_name == "sensor":
        n_side = int(np.ceil(np.sqrt(dim)))
        idx_x = torch.linspace(0, sx - 1, n_side, device=device).round().long()
        idx_y = torch.linspace(0, sy - 1, n_side, device=device).round().long()
        ix_grid, iy_grid = torch.meshgrid(idx_x, idx_y, indexing="ij")
        coords = torch.stack([ix_grid.reshape(-1), iy_grid.reshape(-1)], dim=1)[:dim]
        flat_idx = coords[:, 0] * sy + coords[:, 1]
        phi = torch.zeros((dim, s_total), dtype=dtype, device=device)
        phi[torch.arange(dim, device=device), flat_idx] = 1.0 / max(dxdy, 1e-12)
    else:
        raise ValueError(f"Unknown basis '{basis_name}'.")

    if normalize:
        phi = normalize_basis_rows(phi, dx=dxdy)
    return phi


def measure_with_basis(u: torch.Tensor, psi: torch.Tensor, dx: float) -> torch.Tensor:
    # u: [N, S], psi: [P, S] -> m: [N, P]
    return u @ (psi.transpose(0, 1) * dx)


@dataclass
class ReservoirParams:
    measure_dim: int
    reservoir_dim: int
    washout: int
    leak_alpha: float
    spectral_radius: float
    input_scale: float
    bias_scale: float
    seed: int
    device: torch.device
    dtype: torch.dtype = torch.float32


class FixedReservoirEncoder:
    """Fixed random reservoir encoder: m -> z."""

    def __init__(self, params: ReservoirParams):
        if params.washout <= 0:
            raise ValueError(f"--washout must be >=1, got {params.washout}.")
        if not (0.0 < params.leak_alpha <= 1.0):
            raise ValueError(f"--leak-alpha must be in (0,1], got {params.leak_alpha}.")
        if params.reservoir_dim <= 0:
            raise ValueError(f"--reservoir-dim must be positive, got {params.reservoir_dim}.")
        if params.measure_dim <= 0:
            raise ValueError(f"--measure-dim must be positive, got {params.measure_dim}.")

        self.params = params
        self._init_weights()

    def _init_weights(self) -> None:
        p = self.params
        gen = torch.Generator(device="cpu")
        gen.manual_seed(int(p.seed))

        # W: [M, M], U: [M, P], b: [M]
        W = torch.randn((p.reservoir_dim, p.reservoir_dim), generator=gen, dtype=p.dtype)
        W = W / np.sqrt(max(p.reservoir_dim, 1))
        U = torch.randn((p.reservoir_dim, p.measure_dim), generator=gen, dtype=p.dtype)
        U = (p.input_scale / np.sqrt(max(p.measure_dim, 1))) * U
        b = p.bias_scale * torch.randn((p.reservoir_dim,), generator=gen, dtype=p.dtype)

        current_radius = estimate_spectral_radius(W)
        if current_radius > 0:
            W = W * (p.spectral_radius / current_radius)

        self.W = W.to(p.device)
        self.U = U.to(p.device)
        self.b = b.to(p.device)
        self.init_radius = current_radius
        self.scaled_radius = estimate_spectral_radius(self.W)

    def encode(self, m: torch.Tensor) -> torch.Tensor:
        # m: [N, P] -> z: [N, M]
        p = self.params
        N = m.shape[0]
        r = torch.zeros((N, p.reservoir_dim), dtype=m.dtype, device=m.device)
        for _ in range(p.washout):
            pre = r @ self.W.transpose(0, 1) + m @ self.U.transpose(0, 1) + self.b[None, :]
            r = (1.0 - p.leak_alpha) * r + p.leak_alpha * torch.tanh(pre)
        return r


def ridge_fit_linear_map(x: torch.Tensor, y: torch.Tensor, ridge: float) -> torch.Tensor:
    """Solve y ≈ x @ coef with ridge, returning coef.

    Args:
        x: [N, Din]
        y: [N, Dout]
        ridge: regularization coefficient

    Returns:
        coef: [Din, Dout]
    """
    if ridge < 0:
        raise ValueError(f"ridge must be non-negative, got {ridge}.")
    din = x.shape[1]
    gram = x.transpose(0, 1) @ x
    gram = gram + ridge * torch.eye(din, dtype=x.dtype, device=x.device)
    rhs = x.transpose(0, 1) @ y
    coef = torch.linalg.solve(gram, rhs)
    return coef


def fit_koopman(z0: torch.Tensor, z1: torch.Tensor, ridge_k: float) -> torch.Tensor:
    # z0, z1: [N, M], return Kt: [M, M] for z1_hat = z0 @ Kt
    return ridge_fit_linear_map(z0, z1, ridge=ridge_k)


def stabilize_koopman(kt: torch.Tensor, max_radius: float = 1.0) -> tuple[torch.Tensor, float, float]:
    before = estimate_spectral_radius(kt)
    after = before
    if before > max_radius and before > 0:
        kt = kt * (max_radius / before)
        after = estimate_spectral_radius(kt)
    return kt, before, after


def fit_basis_coefficients(u: torch.Tensor, phi: torch.Tensor, ridge: float) -> torch.Tensor:
    """Project grid function values to basis coefficients.

    Args:
        u: [N, S]
        phi: [Q, S], decoder basis rows
        ridge: regularization for stable projection

    Returns:
        c: [N, Q] so that u ≈ c @ phi
    """
    q = phi.shape[0]
    lhs = phi @ phi.transpose(0, 1)
    lhs = lhs + ridge * torch.eye(q, dtype=u.dtype, device=u.device)
    rhs = phi @ u.transpose(0, 1)  # [Q, N]
    c_t = torch.linalg.solve(lhs, rhs)  # [Q, N]
    return c_t.transpose(0, 1)
