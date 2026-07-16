from __future__ import annotations

import math
import torch

from .target_representation import real_fourier_analysis


def decode_equispaced_point_observation_to_real_fourier(
    features: torch.Tensor,
    q: int,
    *,
    domain_length: float,
) -> torch.Tensor:
    """Decode L2-scaled equispaced observations into real Fourier coefficients.

    Features satisfy ``phi_j=sqrt(L/J) r(x_j)``.  For full observation
    ``J=n_source`` this recovers all retained low source-grid modes.  For
    ``J<n_source``, general high modes may alias under point sampling: this
    decoder is not a pre-observation anti-alias filter.
    """
    if features.ndim < 1 or features.shape[-1] < 2:
        raise ValueError("features must have shape (..., J) with J >= 2")
    J = int(features.shape[-1])
    raw = features * math.sqrt(float(J) / float(domain_length))
    return real_fourier_analysis(raw, q, domain_length=domain_length)
