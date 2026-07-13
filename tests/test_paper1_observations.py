import math

import torch

from pol.paper1.grids import periodic_grid
from pol.paper1.observations import equispaced_periodic_positions, observe_equispaced_periodic


def test_physical_positions_no_endpoint():
    pos = equispaced_periodic_positions(5, 2.0)
    assert torch.allclose(pos, torch.tensor([0.0, 0.4, 0.8, 1.2, 1.6], dtype=torch.float64))
    assert pos[-1] < 2.0


def test_aligned_gather_matches_indices():
    u = torch.arange(24, dtype=torch.float64).reshape(2, 12)
    got = observe_equispaced_periodic(u, 4, domain_length=1.0, l2_scale=False)
    assert torch.equal(got, u[:, [0, 3, 6, 9]])


def test_non_aligned_fourier_evaluation():
    n, J = 10, 6
    x = periodic_grid(n, 1.0)
    u = torch.cos(2.0 * torch.pi * 3.0 * x) - 0.3 * torch.sin(2.0 * torch.pi * 2.0 * x)
    got = observe_equispaced_periodic(u, J, domain_length=1.0, l2_scale=False)
    xp = equispaced_periodic_positions(J, 1.0)
    expected = torch.cos(2.0 * torch.pi * 3.0 * xp) - 0.3 * torch.sin(2.0 * torch.pi * 2.0 * xp)
    assert torch.allclose(got, expected, atol=1e-10)


def test_l2_scaled_constant_norm():
    obs = observe_equispaced_periodic(3.0 * torch.ones(7, dtype=torch.float64), 5, domain_length=2.0)
    assert torch.linalg.norm(obs).item() == pytest_approx(3.0 * math.sqrt(2.0))


def pytest_approx(value):
    import pytest

    return pytest.approx(value, abs=1e-12, rel=1e-12)


def test_batch_dtype():
    u = torch.ones(3, 9, dtype=torch.float32)
    out = observe_equispaced_periodic(u, 4, domain_length=1.0, l2_scale=False)
    assert out.shape == (3, 4)
    assert out.dtype == torch.float32
