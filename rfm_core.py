"""Core utilities for Random Feature Models (RFM) in POL_for_PDE.

This module implements:
  - Gram matrix accumulation for RFM ridge regression
  - Closed-form solve for feature weights
  - Prediction helper for batched inference
  - Lightweight model serialization helpers
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Optional, Tuple

import torch


FeatureFn = Callable[[torch.Tensor], torch.Tensor]


@dataclass
class RFMModel:
    """Container for a fitted RFM."""

    alpha: torch.Tensor
    m: int
    feature_fn: FeatureFn

    def predict(self, a_batch: torch.Tensor) -> torch.Tensor:
        """Predict output for a batch of inputs."""
        phi = self.feature_fn(a_batch)
        return predict_from_features(self.alpha, phi, self.m)


def accumulate_gram_and_b(
    phi: torch.Tensor,
    y: torch.Tensor,
    gram: torch.Tensor,
    b: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Accumulate Gram matrix and right-hand side for RFM regression.

    Args:
        phi: (B, m, K) random feature outputs for a batch.
        y: (B, K) target output for a batch.
        gram: (m, m) accumulator for sum_i <phi_i, phi_i>.
        b: (m,) accumulator for sum_i <phi_i, y_i>.
    """
    k = phi.shape[-1]
    gram += torch.einsum("bmk,bnk->mn", phi, phi) / k
    b += torch.einsum("bmk,bk->m", phi, y) / k
    return gram, b


def solve_alpha(gram: torch.Tensor, b: torch.Tensor, m: int, lam: float) -> torch.Tensor:
    """Solve the RFM normal equations for alpha."""
    device = gram.device
    dtype = gram.dtype
    eye = torch.eye(m, device=device, dtype=dtype)
    gram_reg = gram / m + lam * eye
    linalg_error = getattr(torch.linalg, "LinAlgError", RuntimeError)
    if lam > 0:
        try:
            return torch.linalg.solve(gram_reg, b)
        except (RuntimeError, linalg_error):
            pass
    jitter_levels = [1e-10, 1e-8, 1e-6, 1e-4]
    for jitter in jitter_levels:
        try:
            return torch.linalg.solve(gram_reg + jitter * eye, b)
        except (RuntimeError, linalg_error):
            continue
    return torch.linalg.pinv(gram_reg) @ b


def fit_rfm(
    data_loader: Iterable[Tuple[torch.Tensor, torch.Tensor]],
    feature_fn: FeatureFn,
    m: int,
    lam: float,
    device: torch.device,
) -> torch.Tensor:
    """Fit an RFM model by accumulating Gram matrix and solving for alpha."""
    gram = torch.zeros((m, m), device=device, dtype=torch.float64)
    b = torch.zeros((m,), device=device, dtype=torch.float64)

    for a, y in data_loader:
        a = a.to(device)
        y = y.to(device)
        if y.ndim > 2:
            y = y.reshape(y.shape[0], -1)
        if a.ndim > 2:
            a = a.reshape(a.shape[0], *a.shape[1:])

        with torch.no_grad():
            phi = feature_fn(a).to(torch.float64)
            if phi.ndim > 3:
                phi = phi.reshape(phi.shape[0], phi.shape[1], -1)
        gram, b = accumulate_gram_and_b(phi, y.to(torch.float64), gram, b)

    alpha = solve_alpha(gram, b, m, lam).to(torch.float32)
    return alpha


def predict_from_features(alpha: torch.Tensor, phi: torch.Tensor, m: int) -> torch.Tensor:
    """Compute RFM predictions from precomputed features."""
    return torch.einsum("m,bmk->bk", alpha, phi) / m


def save_rfm_model(
    path: str,
    *,
    alpha: torch.Tensor,
    theta: Dict[str, torch.Tensor],
    hyperparams: Dict[str, float],
) -> None:
    """Save RFM model state to disk."""
    payload = {
        "alpha": alpha.detach().cpu(),
        "theta": {key: value.detach().cpu() for key, value in theta.items()},
        "hyperparams": hyperparams,
    }
    torch.save(payload, path)


def load_rfm_model(path: str) -> Dict[str, object]:
    """Load RFM model state from disk."""
    return torch.load(path, map_location="cpu")
