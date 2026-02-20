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
def fit_ridge_streaming_standardized(
    dataloader,
    feature_fn: Callable[[torch.Tensor], torch.Tensor],
    ridge_lambda: float,
    *,
    dtype: torch.dtype = torch.float64,
    regularize_bias: bool = False,
    eps: float = 1e-6,
) -> Dict[str, torch.Tensor]:
    if ridge_lambda < 0.0:
        raise ValueError("ridge_lambda must be non-negative")
    if eps <= 0.0:
        raise ValueError("eps must be positive")

    gram = None
    cross = None

    for x_batch, y_batch in dataloader:
        phi = feature_fn(x_batch).to(dtype=dtype)
        y = y_batch.to(dtype=dtype, device=phi.device)
        x_aug = _append_bias(phi)

        if gram is None:
            d_aug = x_aug.shape[1]
            out_dim = y.shape[1]
            gram = torch.zeros((d_aug, d_aug), dtype=dtype, device=phi.device)
            cross = torch.zeros((d_aug, out_dim), dtype=dtype, device=phi.device)

        gram += x_aug.t() @ x_aug
        cross += x_aug.t() @ y

    if gram is None or cross is None:
        raise ValueError("empty dataloader")

    d_aug = gram.shape[0]
    d = d_aug - 1
    gram_ff = gram[:d, :d]
    sum_phi = gram[:d, -1]
    n = gram[-1, -1]
    if n <= 0:
        raise ValueError("invalid sample count accumulated in Gram matrix")

    mean = sum_phi / n
    sum_sq = torch.diagonal(gram_ff)
    var = torch.clamp(sum_sq / n - mean.pow(2), min=0.0)
    std = torch.sqrt(var)
    std_eps = std + eps
    inv_std = 1.0 / std_eps

    gram_center = gram_ff - torch.outer(sum_phi, sum_phi) / n
    gram_scaled = inv_std[:, None] * gram_center * inv_std[None, :]
    gram_std = torch.zeros_like(gram)
    gram_std[:d, :d] = gram_scaled
    gram_std[-1, -1] = n

    cross_f = cross[:d, :]
    cross_b = cross[-1, :]
    cross_center = cross_f - mean[:, None] * cross_b[None, :]
    cross_scaled = inv_std[:, None] * cross_center
    cross_std = torch.zeros_like(cross)
    cross_std[:d, :] = cross_scaled
    cross_std[-1, :] = cross_b

    eye = torch.eye(d_aug, device=gram.device, dtype=gram.dtype)
    if not regularize_bias:
        eye[-1, -1] = 0.0

    reg_gram_std = gram_std + ridge_lambda * eye
    chol_std = torch.linalg.cholesky(reg_gram_std)
    w_std = torch.cholesky_solve(cross_std, chol_std)

    w_feat_std = w_std[:d, :]
    w_bias_std = w_std[-1:, :]
    mean_scaled = (mean * inv_std).unsqueeze(0)
    w_feat_raw = inv_std[:, None] * w_feat_std
    w_bias_raw = w_bias_std - mean_scaled @ w_feat_std
    w_raw = torch.cat([w_feat_raw, w_bias_raw], dim=0)

    return {
        "W": w_raw,
        "W_std": w_std,
        "mean": mean,
        "std": std,
        "gram": gram,
        "cross": cross,
        "gram_std": gram_std,
        "cross_std": cross_std,
        "eps": torch.tensor(eps, device=gram.device, dtype=gram.dtype),
    }


@torch.no_grad()
def predict_linear(features: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
    x_aug = _append_bias(features)
    return x_aug @ W
