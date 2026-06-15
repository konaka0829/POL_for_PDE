from __future__ import annotations

import torch


def _grid_spacing(values: torch.Tensor, *, domain_length: float) -> float:
    if values.shape[-1] <= 0:
        raise ValueError("spatial axis must be non-empty")
    return float(domain_length) / float(values.shape[-1])


def discrete_l2h_norm(values: torch.Tensor, *, domain_length: float = 1.0) -> torch.Tensor:
    h = _grid_spacing(values, domain_length=domain_length)
    return torch.sqrt(h * torch.sum(values * values, dim=-1))


def per_sample_abs_l2h_error(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    domain_length: float = 1.0,
) -> torch.Tensor:
    if pred.shape != target.shape:
        raise ValueError(f"pred and target must have the same shape, got {pred.shape} and {target.shape}")
    return discrete_l2h_norm(pred - target, domain_length=domain_length)


def dataset_abs_l2h_error(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    domain_length: float = 1.0,
) -> float:
    return dataset_abs_l2h_rmse(pred, target, domain_length=domain_length)


def dataset_abs_l2h_rmse(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    domain_length: float = 1.0,
) -> float:
    per_sample = per_sample_abs_l2h_error(pred, target, domain_length=domain_length)
    return torch.sqrt(torch.mean(per_sample * per_sample)).item()


def per_sample_rel_l2h_error(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    domain_length: float = 1.0,
    eps: float = 1e-12,
) -> torch.Tensor:
    numer = per_sample_abs_l2h_error(pred, target, domain_length=domain_length)
    denom = discrete_l2h_norm(target, domain_length=domain_length)
    return numer / (denom + float(eps))


def dataset_rel_l2h_mean(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    domain_length: float = 1.0,
    eps: float = 1e-12,
) -> float:
    per_sample = per_sample_rel_l2h_error(pred, target, domain_length=domain_length, eps=eps)
    return torch.mean(per_sample).item()


def dataset_rel_l2h_aggregate(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    domain_length: float = 1.0,
    eps: float = 1e-12,
) -> float:
    numer = per_sample_abs_l2h_error(pred, target, domain_length=domain_length)
    denom = discrete_l2h_norm(target, domain_length=domain_length)
    return (
        torch.sqrt(torch.sum(numer * numer))
        / torch.sqrt(torch.sum(denom * denom) + float(eps))
    ).item()


def rms_l2(pred: torch.Tensor, target: torch.Tensor) -> float:
    return dataset_abs_l2h_error(pred, target)
