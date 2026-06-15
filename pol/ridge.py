from __future__ import annotations

from typing import Callable, Dict

import torch
import math


def _append_bias(x: torch.Tensor) -> torch.Tensor:
    ones = torch.ones((x.shape[0], 1), device=x.device, dtype=x.dtype)
    return torch.cat([x, ones], dim=-1)


RIDGE_CONVENTION = "normalized_empirical_l2h_unweighted_frobenius"


def _ridge_diagnostics(
    *,
    W: torch.Tensor,
    gram_features: torch.Tensor,
    ridge_zeta: float,
    dx: float,
    n_samples: int,
) -> Dict[str, torch.Tensor]:
    zeta_eff = float(ridge_zeta) / float(dx)
    C = gram_features / float(n_samples)
    eigvals = torch.linalg.eigvalsh(C).clamp_min(0.0)
    d_eff = torch.sum(eigvals / (eigvals + zeta_eff)).to(dtype=W.dtype)
    cond = ((eigvals.max() + zeta_eff) / (eigvals.min() + zeta_eff)).to(dtype=W.dtype)
    return {
        "W_fro_norm": torch.linalg.matrix_norm(W[:-1, :], ord="fro"),
        "W_hs_norm_l2h": math.sqrt(float(dx)) * torch.linalg.matrix_norm(W[:-1, :], ord="fro"),
        "d_eff": d_eff,
        "cond_zeta": cond,
        "ridge_zeta": torch.tensor(float(ridge_zeta), dtype=W.dtype, device=W.device),
        "ridge_zeta_eff": torch.tensor(zeta_eff, dtype=W.dtype, device=W.device),
        "dx": torch.tensor(float(dx), dtype=W.dtype, device=W.device),
        "num_train_samples": torch.tensor(int(n_samples), dtype=torch.int64, device=W.device),
        "effective_code_lambda_legacy_equivalent": torch.tensor(
            float(n_samples) * float(ridge_zeta) / float(dx),
            dtype=W.dtype,
            device=W.device,
        ),
    }


@torch.no_grad()
def fit_ridge_streaming(
    dataloader,
    feature_fn: Callable[[torch.Tensor], torch.Tensor],
    ridge_lambda: float,
    *,
    dtype: torch.dtype = torch.float64,
    regularize_bias: bool = False,
    ridge_zeta: float | None = None,
    dx: float = 1.0,
    convention: str = RIDGE_CONVENTION,
    progress_fn: Callable[[int, int | None], None] | None = None,
) -> Dict[str, torch.Tensor]:
    if ridge_zeta is None:
        ridge_zeta = ridge_lambda
    if ridge_zeta < 0.0:
        raise ValueError("ridge_zeta must be non-negative")
    if dx <= 0.0:
        raise ValueError("dx must be positive")

    gram = None
    cross = None
    x_aug_batches = [] if ridge_zeta == 0.0 else None
    y_batches = [] if ridge_zeta == 0.0 else None

    n_samples = 0
    total_batches = len(dataloader) if hasattr(dataloader, "__len__") else None
    for batch_idx, (x_batch, y_batch) in enumerate(dataloader, start=1):
        phi = feature_fn(x_batch).to(dtype=dtype)
        y = y_batch.to(dtype=dtype, device=phi.device)
        x_aug = _append_bias(phi)
        n_samples += int(x_aug.shape[0])
        if x_aug_batches is not None and y_batches is not None:
            x_aug_batches.append(x_aug)
            y_batches.append(y)

        if gram is None:
            d = x_aug.shape[1]
            out_dim = y.shape[1]
            gram = torch.zeros((d, d), dtype=dtype, device=phi.device)
            cross = torch.zeros((d, out_dim), dtype=dtype, device=phi.device)

        gram += x_aug.t() @ x_aug
        cross += x_aug.t() @ y
        if progress_fn is not None:
            progress_fn(batch_idx, total_batches)

    if gram is None or cross is None:
        raise ValueError("empty dataloader")

    d = gram.shape[0]
    eye = torch.eye(d, device=gram.device, dtype=gram.dtype)
    if not regularize_bias:
        eye[-1, -1] = 0.0
    if convention == RIDGE_CONVENTION:
        reg_weight = float(n_samples) * float(ridge_zeta) / float(dx)
    elif convention == "legacy_unnormalized_gram":
        reg_weight = float(ridge_zeta)
    else:
        raise ValueError(f"unsupported ridge convention: {convention}")
    reg_gram = gram + reg_weight * eye

    if ridge_zeta == 0.0 and x_aug_batches is not None and y_batches is not None:
        x_full = torch.cat(x_aug_batches, dim=0)
        y_full = torch.cat(y_batches, dim=0)
        w = torch.linalg.lstsq(x_full, y_full).solution
    else:
        try:
            chol = torch.linalg.cholesky(reg_gram)
            w = torch.cholesky_solve(cross, chol)
        except RuntimeError:
            w = torch.linalg.lstsq(reg_gram, cross).solution
    result = {
        "W": w,
        "gram": gram,
        "cross": cross,
        "ridge_convention": convention,
    }
    result.update(
        _ridge_diagnostics(
            W=w,
            gram_features=gram[:-1, :-1],
            ridge_zeta=float(ridge_zeta),
            dx=float(dx),
            n_samples=n_samples,
        )
    )
    return result


@torch.no_grad()
def fit_ridge_streaming_standardized(
    dataloader,
    feature_fn: Callable[[torch.Tensor], torch.Tensor],
    ridge_lambda: float,
    *,
    dtype: torch.dtype = torch.float64,
    regularize_bias: bool = False,
    ridge_zeta: float | None = None,
    dx: float = 1.0,
    convention: str = RIDGE_CONVENTION,
    eps: float = 1e-6,
    progress_fn: Callable[[int, int | None], None] | None = None,
) -> Dict[str, torch.Tensor]:
    if ridge_zeta is None:
        ridge_zeta = ridge_lambda
    if ridge_zeta < 0.0:
        raise ValueError("ridge_zeta must be non-negative")
    if eps <= 0.0:
        raise ValueError("eps must be positive")

    gram = None
    cross = None

    n_samples = 0
    total_batches = len(dataloader) if hasattr(dataloader, "__len__") else None
    for batch_idx, (x_batch, y_batch) in enumerate(dataloader, start=1):
        phi = feature_fn(x_batch).to(dtype=dtype)
        y = y_batch.to(dtype=dtype, device=phi.device)
        x_aug = _append_bias(phi)
        n_samples += int(x_aug.shape[0])

        if gram is None:
            d_aug = x_aug.shape[1]
            out_dim = y.shape[1]
            gram = torch.zeros((d_aug, d_aug), dtype=dtype, device=phi.device)
            cross = torch.zeros((d_aug, out_dim), dtype=dtype, device=phi.device)

        gram += x_aug.t() @ x_aug
        cross += x_aug.t() @ y
        if progress_fn is not None:
            progress_fn(batch_idx, total_batches)

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

    if convention == RIDGE_CONVENTION:
        reg_weight = float(n_samples) * float(ridge_zeta) / float(dx)
    elif convention == "legacy_unnormalized_gram":
        reg_weight = float(ridge_zeta)
    else:
        raise ValueError(f"unsupported ridge convention: {convention}")
    reg_gram_std = gram_std + reg_weight * eye
    try:
        chol_std = torch.linalg.cholesky(reg_gram_std)
        w_std = torch.cholesky_solve(cross_std, chol_std)
    except RuntimeError:
        w_std = torch.linalg.lstsq(reg_gram_std, cross_std).solution

    w_feat_std = w_std[:d, :]
    w_bias_std = w_std[-1:, :]
    mean_scaled = (mean * inv_std).unsqueeze(0)
    w_feat_raw = inv_std[:, None] * w_feat_std
    w_bias_raw = w_bias_std - mean_scaled @ w_feat_std
    w_raw = torch.cat([w_feat_raw, w_bias_raw], dim=0)

    result = {
        "W": w_raw,
        "W_std": w_std,
        "mean": mean,
        "std": std,
        "gram": gram,
        "cross": cross,
        "gram_std": gram_std,
        "cross_std": cross_std,
        "eps": torch.tensor(eps, device=gram.device, dtype=gram.dtype),
        "ridge_convention": convention,
    }
    result.update(
        _ridge_diagnostics(
            W=w_raw,
            gram_features=gram[:-1, :-1],
            ridge_zeta=float(ridge_zeta),
            dx=float(dx),
            n_samples=n_samples,
        )
    )
    return result


@torch.no_grad()
def fit_ridge_from_tensors(
    features: torch.Tensor,
    targets: torch.Tensor,
    *,
    ridge_zeta: float,
    dx: float,
    dtype: torch.dtype = torch.float64,
    regularize_bias: bool = False,
    convention: str = RIDGE_CONVENTION,
) -> Dict[str, torch.Tensor]:
    if features.ndim != 2 or targets.ndim != 2:
        raise ValueError("features and targets must be rank-2 tensors")
    if features.shape[0] != targets.shape[0]:
        raise ValueError("features and targets must have the same number of samples")
    if ridge_zeta < 0.0:
        raise ValueError("ridge_zeta must be non-negative")
    if dx <= 0.0:
        raise ValueError("dx must be positive")

    phi = features.to(dtype=dtype)
    y = targets.to(device=phi.device, dtype=dtype)
    x_aug = _append_bias(phi)
    gram = x_aug.t() @ x_aug
    cross = x_aug.t() @ y
    n_samples = int(phi.shape[0])
    d = gram.shape[0]
    eye = torch.eye(d, device=gram.device, dtype=gram.dtype)
    if not regularize_bias:
        eye[-1, -1] = 0.0
    if convention == RIDGE_CONVENTION:
        reg_weight = float(n_samples) * float(ridge_zeta) / float(dx)
    elif convention == "legacy_unnormalized_gram":
        reg_weight = float(ridge_zeta)
    else:
        raise ValueError(f"unsupported ridge convention: {convention}")
    reg_gram = gram + reg_weight * eye
    if ridge_zeta == 0.0:
        w = torch.linalg.lstsq(x_aug, y).solution
    else:
        try:
            chol = torch.linalg.cholesky(reg_gram)
            w = torch.cholesky_solve(cross, chol)
        except RuntimeError:
            w = torch.linalg.lstsq(reg_gram, cross).solution
    result = {
        "W": w,
        "gram": gram,
        "cross": cross,
        "ridge_convention": convention,
    }
    result.update(
        _ridge_diagnostics(
            W=w,
            gram_features=gram[:-1, :-1],
            ridge_zeta=float(ridge_zeta),
            dx=float(dx),
            n_samples=n_samples,
        )
    )
    return result


@torch.no_grad()
def predict_linear(features: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
    x_aug = _append_bias(features)
    return x_aug @ W
