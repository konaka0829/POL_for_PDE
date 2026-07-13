import math

import pytest
import torch

from pol.paper1.grids import periodic_grid
from pol.paper1.target_representation import real_fourier_analysis, real_fourier_synthesis, validate_real_fourier_dim


def test_coefficient_ordering_and_analytic_values():
    L = 2.0
    nx = 64
    x = periodic_grid(nx, L)
    f = 3.0 + 2.5 * torch.cos(2.0 * torch.pi * x / L) - 1.25 * torch.sin(4.0 * torch.pi * x / L)
    c = real_fourier_analysis(f, 5, domain_length=L)
    expected = torch.tensor(
        [3.0 * math.sqrt(L), 2.5 * math.sqrt(L / 2.0), 0.0, 0.0, -1.25 * math.sqrt(L / 2.0)],
        dtype=torch.float64,
    )
    assert torch.allclose(c, expected, atol=1e-10)


def test_analysis_synthesis_identity_on_retained_subspace():
    coeffs = torch.tensor([[1.0, 0.5, -0.25, 0.1, 0.2]], dtype=torch.float64)
    values = real_fourier_synthesis(coeffs, 33, domain_length=1.0)
    got = real_fourier_analysis(values, 5, domain_length=1.0)
    assert torch.allclose(got, coeffs, atol=1e-10)


def test_parseval_retained_subspace():
    coeffs = torch.tensor([1.0, 0.5, -0.25, 0.1, 0.2], dtype=torch.float64)
    values = real_fourier_synthesis(coeffs, 64, domain_length=2.0)
    dx = 2.0 / 64.0
    assert torch.allclose(dx * torch.sum(values * values), torch.sum(coeffs * coeffs), atol=1e-10)


def test_constant_only_coefficient():
    f = 4.0 * torch.ones(2, 16, dtype=torch.float32)
    c = real_fourier_analysis(f, 3, domain_length=1.0)
    assert c.dtype == torch.float32
    assert torch.allclose(c[:, 0], torch.full((2,), 4.0, dtype=torch.float32), atol=1e-6)
    assert torch.allclose(c[:, 1:], torch.zeros(2, 2, dtype=torch.float32), atol=1e-6)


def test_invalid_q():
    with pytest.raises(ValueError):
        validate_real_fourier_dim(4, 16)
    with pytest.raises(ValueError):
        validate_real_fourier_dim(17, 16)
    with pytest.raises(ValueError):
        validate_real_fourier_dim(3, 1)
