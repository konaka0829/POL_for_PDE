from __future__ import annotations

from typing import Callable, Dict, Tuple

import torch


def _append_bias(x: torch.Tensor) -> torch.Tensor:
    ones = torch.ones((x.shape[0], 1), device=x.device, dtype=x.dtype)
    return torch.cat([x, ones], dim=-1)


@torch.no_grad()
def fit_ridge_streaming(
    dataloader,
    feature_fn: Callable[[torch.Tensor], torch.Tensor],
    ridge_lambda: float,
    *,
    dtype: torch.dtype = torch.float64,
    regularize_bias: bool = False,
) -> Dict[str, torch.Tensor]:
    if ridge_lambda < 0.0:
        raise ValueError("ridge_lambda must be non-negative")

    gram = None
    cross = None

    for x_batch, y_batch in dataloader:
        phi = feature_fn(x_batch).to(dtype=dtype)
        y = y_batch.to(dtype=dtype, device=phi.device)
        x_aug = _append_bias(phi)

        if gram is None:
            d = x_aug.shape[1]
            out_dim = y.shape[1]
            gram = torch.zeros((d, d), dtype=dtype, device=phi.device)
            cross = torch.zeros((d, out_dim), dtype=dtype, device=phi.device)

        gram += x_aug.t() @ x_aug
        cross += x_aug.t() @ y

    if gram is None or cross is None:
        raise ValueError("empty dataloader")

    d = gram.shape[0]
    eye = torch.eye(d, device=gram.device, dtype=gram.dtype)
    if not regularize_bias:
        eye[-1, -1] = 0.0
    reg_gram = gram + ridge_lambda * eye

    chol = torch.linalg.cholesky(reg_gram)
    w = torch.cholesky_solve(cross, chol)
    return {
        "W": w,
        "gram": gram,
        "cross": cross,
    }


@torch.no_grad()
def predict_linear(features: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
    x_aug = _append_bias(features)
    return x_aug @ W
