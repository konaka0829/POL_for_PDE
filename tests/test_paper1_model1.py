import pytest
import torch

from pol.paper1.grids import periodic_grid
from pol.paper1.model1 import decode_equispaced_point_observation_to_real_fourier
from pol.paper1.observations import observe_equispaced_periodic
from pol.paper1.target_representation import real_fourier_analysis


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_model1_scaling_order_batch_full_and_reduced(dtype):
    x = periodic_grid(32, 2.0, dtype=dtype)
    fields = torch.stack([0.3 + 0.7 * torch.cos(torch.pi * 2 * x) - 0.2 * torch.sin(torch.pi * 3 * x), torch.cos(torch.pi * x)])
    features = observe_equispaced_periodic(fields, 32, domain_length=2.0, l2_scale=True)
    got = decode_equispaced_point_observation_to_real_fourier(features, 9, domain_length=2.0)
    expected = real_fourier_analysis(fields, 9, domain_length=2.0)
    assert torch.allclose(got, expected, atol=1e-5 if dtype == torch.float32 else 1e-11)
    reduced = decode_equispaced_point_observation_to_real_fourier(observe_equispaced_periodic(fields, 16, domain_length=2.0, l2_scale=True), 9, domain_length=2.0)
    assert torch.allclose(reduced, expected, atol=1e-5 if dtype == torch.float32 else 1e-11)


def test_model1_high_mode_alias_counterexample():
    x = periodic_grid(64, 1.0)
    low = torch.cos(4 * torch.pi * x); high = low + 0.4 * torch.cos(36 * torch.pi * x)
    truth = real_fourier_analysis(low.unsqueeze(0), 7, domain_length=1.0)
    decoded = decode_equispaced_point_observation_to_real_fourier(observe_equispaced_periodic(high.unsqueeze(0), 16, domain_length=1.0), 7, domain_length=1.0)
    assert not torch.allclose(decoded, truth, atol=1e-10)
