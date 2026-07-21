from __future__ import annotations

from dataclasses import dataclass
import torch

from .target_representation import real_fourier_analysis, real_fourier_synthesis


def l2_analysis_matrix(q: int, J: int, *, domain_length: float, dtype: torch.dtype = torch.float64, device: torch.device | str = "cpu") -> torch.Tensor:
    """D mapping column L2-scaled point features to real Fourier coefficients."""
    eye = torch.eye(J, dtype=dtype, device=device)
    scaled_points = eye / (domain_length / J) ** 0.5
    return real_fourier_analysis(scaled_points, q, domain_length=domain_length).T.contiguous()


def l2_synthesis_matrix(q: int, J: int, *, domain_length: float, dtype: torch.dtype = torch.float64, device: torch.device | str = "cpu") -> torch.Tensor:
    """S mapping column real coefficients to L2-scaled point features."""
    eye = torch.eye(q, dtype=dtype, device=device)
    points = real_fourier_synthesis(eye, J, domain_length=domain_length).T
    return points * (domain_length / J) ** 0.5


@dataclass(frozen=True)
class AffineReadout:
    # Prediction uses row samples: y = x @ W.T + b; mathematical W is (q,J).
    W: torch.Tensor
    b: torch.Tensor

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.W.T + self.b


def fit_centered_affine_ridge(x: torch.Tensor, y: torch.Tensor, zeta: float) -> AffineReadout:
    if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0] or x.shape[0] == 0:
        raise ValueError("ridge expects x=(N,J), y=(N,q) with shared positive N")
    if x.dtype != y.dtype or x.device != y.device or zeta < 0:
        raise ValueError("ridge dtype/device mismatch or negative zeta")
    xm, ym = x.mean(0), y.mean(0)
    xc, yc = x - xm, y - ym
    if zeta == 0:
        beta = torch.linalg.lstsq(xc, yc).solution
    elif x.shape[1] <= x.shape[0]:
        gram = xc.T @ xc / x.shape[0]
        rhs = xc.T @ yc / x.shape[0]
        beta = torch.linalg.solve(gram + zeta * torch.eye(x.shape[1], dtype=x.dtype, device=x.device), rhs)
    else:
        gram = xc @ xc.T / x.shape[0]
        dual = torch.linalg.solve(gram + zeta * torch.eye(x.shape[0], dtype=x.dtype, device=x.device), yc / x.shape[0])
        beta = xc.T @ dual
    W = beta.T.contiguous()
    return AffineReadout(W, ym - xm @ beta)
