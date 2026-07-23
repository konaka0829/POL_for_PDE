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
    """Decode L2-scaled point values with a fixed Fourier decoder.

    Input features satisfy ``phi_j=sqrt(L/J) r(x_j)``; the scaling is undone
    before applying the fixed decoder.  For ``J<n_source`` general fields may
    contain high modes that alias under point sampling: this decoder is not a
    pre-observation anti-alias filter.  Only ordinary low Fourier modes that
    are uniquely represented by the observation grid are copied in the
    standard constant/cos/sin ordering.  Requested unavailable target modes
    are zero padded; no learned high-mode extrapolation is performed.
    """
    if features.ndim < 1 or features.shape[-1] < 2:
        raise ValueError("features must have shape (..., J) with J >= 2")
    J = int(features.shape[-1])
    raw = features * math.sqrt(float(J) / float(domain_length))
    q_observable = J if J % 2 else J - 1
    retained = min(q, q_observable)
    decoded = real_fourier_analysis(raw, retained, domain_length=domain_length)
    if retained == q:
        return decoded
    return torch.cat(
        [decoded, torch.zeros((*decoded.shape[:-1], q - retained), dtype=decoded.dtype, device=decoded.device)],
        dim=-1,
    )
