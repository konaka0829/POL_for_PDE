from __future__ import annotations

from dataclasses import dataclass

import torch

from .grids import spectral_resample_periodic
from .target_representation import real_fourier_analysis


@dataclass(frozen=True)
class Paper1FiniteData:
    sample_ids: torch.Tensor
    u0_data: torch.Tensor
    y_target_data: torch.Tensor
    target_coefficients: torch.Tensor | None
    metadata: dict[str, object]


@torch.no_grad()
def derive_finite_resolution_data(
    u0_reference: torch.Tensor,
    y_reference: torch.Tensor,
    *,
    target_data_nx: int,
    target_output_dim: int | None,
    domain_length: float,
    sample_ids: torch.Tensor | None = None,
) -> Paper1FiniteData:
    """Low-pass and resynthesize reference input/label fields at ``n_tar``."""
    if u0_reference.shape != y_reference.shape or u0_reference.ndim != 2:
        raise ValueError("u0_reference and y_reference must share shape (samples, reference_nx)")
    n = u0_reference.shape[0]
    ids = torch.arange(n, device=u0_reference.device) if sample_ids is None else sample_ids
    if ids.ndim != 1 or ids.numel() != n:
        raise ValueError("sample_ids must have one entry per sample")
    u = spectral_resample_periodic(u0_reference, target_data_nx, domain_length=domain_length)
    y = spectral_resample_periodic(y_reference, target_data_nx, domain_length=domain_length)
    coeff = None if target_output_dim is None else real_fourier_analysis(y, target_output_dim, domain_length=domain_length)
    return Paper1FiniteData(
        sample_ids=ids.clone(), u0_data=u, y_target_data=y, target_coefficients=coeff,
        metadata={"schema_version": "paper1-finite-data-v1", "reference_nx": int(u0_reference.shape[-1]), "target_data_nx": int(target_data_nx), "target_output_dim": target_output_dim, "domain_length": float(domain_length)},
    )


@torch.no_grad()
def build_surrogate_initial_state(
    u0_data: torch.Tensor,
    *,
    surrogate_internal_nx: int,
    domain_length: float,
) -> torch.Tensor:
    """Evaluate only the finite-input trigonometric interpolant at ``n_sur`` nodes.

    No reference/master data are accepted, so discarded high modes cannot be
    reintroduced through this interface.
    """
    return spectral_resample_periodic(u0_data, surrogate_internal_nx, domain_length=domain_length)
