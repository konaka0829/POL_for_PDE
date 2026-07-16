from __future__ import annotations

import torch

from .grids import spectral_resample_periodic


def periodic_l2_norm(values: torch.Tensor, *, domain_length: float) -> torch.Tensor:
    """Periodic trapezoidal L2 norm over the last, endpoint-free grid axis."""
    if values.ndim < 1 or values.shape[-1] < 2:
        raise ValueError("values must have shape (..., nx), nx >= 2")
    return torch.sqrt((float(domain_length) / values.shape[-1]) * torch.sum(torch.abs(values) ** 2, dim=-1))


def samplewise_l2_errors(prediction: torch.Tensor, reference: torch.Tensor, *, domain_length: float, relative_epsilon: float = 1e-14) -> dict[str, torch.Tensor]:
    """Return absolute and safe relative sample-wise L2 errors."""
    if prediction.shape != reference.shape:
        raise ValueError("prediction and reference must have identical shapes")
    absolute = periodic_l2_norm(prediction - reference, domain_length=domain_length)
    denominator = periodic_l2_norm(reference, domain_length=domain_length)
    relative = torch.where(denominator > relative_epsilon, absolute / denominator, torch.where(absolute <= relative_epsilon, torch.zeros_like(absolute), absolute / relative_epsilon))
    return {"absolute": absolute, "relative": relative, "reference_norm": denominator}


def aggregate_errors(samplewise: torch.Tensor) -> dict[str, float]:
    """Aggregate finite sample-wise values as mean, median, and max."""
    if samplewise.numel() == 0 or not bool(torch.isfinite(samplewise).all()):
        raise ValueError("samplewise errors must be nonempty and finite")
    x = samplewise.detach().cpu().to(torch.float64)
    return {"mean": float(x.mean()), "median": float(x.median()), "max": float(x.max())}


def compare_fields_on_common_grid(candidate: torch.Tensor, reference: torch.Tensor, *, common_nx: int, domain_length: float) -> dict[str, object]:
    """Spectrally reconstruct two periodic batches before comparing them."""
    c = spectral_resample_periodic(candidate, common_nx, domain_length=domain_length)
    r = spectral_resample_periodic(reference, common_nx, domain_length=domain_length)
    errors = samplewise_l2_errors(c, r, domain_length=domain_length)
    return {"absolute": errors["absolute"], "relative": errors["relative"], "absolute_aggregate": aggregate_errors(errors["absolute"]), "relative_aggregate": aggregate_errors(errors["relative"])}
