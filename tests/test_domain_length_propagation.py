import math

import pytest
import torch

from pol.model123_1d.feature_cache import make_feature_cache_key
from pol.model123_1d.metrics import discrete_l2h_norm
from pol.model123_1d.error_decomposition import spectral_derivatives_1d
from pol.reservoir_1d import Reservoir1DSolver, ReservoirConfig


def test_heat_reservoir_single_mode_uses_physical_wavenumber():
    s = 64
    L = 2.0
    mode = 2
    nu = 0.03
    t = 0.2
    dt = 0.1
    x = torch.arange(s, dtype=torch.float64) * (L / s)
    z0 = torch.sin(2.0 * math.pi * mode * x / L).unsqueeze(0)
    solver = Reservoir1DSolver(ReservoirConfig(reservoir="heat", heat_nu=nu, domain_length=L))
    out = solver.simulate(z0, dt=dt, Tr=t, obs_steps=[int(round(t / dt))])[-1]
    amp = math.exp(-nu * (2.0 * math.pi * mode / L) ** 2 * t)
    assert torch.allclose(out, amp * z0, atol=1e-7, rtol=1e-7)


def test_advection_reservoir_single_mode_uses_physical_shift():
    s = 64
    L = 2.0
    mode = 1
    c = 0.25
    t = 0.2
    dt = 0.1
    x = torch.arange(s, dtype=torch.float64) * (L / s)
    z0 = torch.sin(2.0 * math.pi * mode * x / L).unsqueeze(0)
    solver = Reservoir1DSolver(ReservoirConfig(reservoir="advection", advection_c=c, domain_length=L))
    out = solver.simulate(z0, dt=dt, Tr=t, obs_steps=[int(round(t / dt))])[-1]
    expected = torch.sin(2.0 * math.pi * mode * (x - c * t) / L).unsqueeze(0)
    assert torch.allclose(out, expected, atol=1e-7, rtol=1e-7)


def test_l2h_norm_scales_with_dx_domain_length():
    values = torch.ones(1, 8, dtype=torch.float64)
    assert discrete_l2h_norm(values, domain_length=2.0).item() == pytest.approx(math.sqrt(2.0))


def test_spectral_derivative_single_mode_amplitude_uses_domain_length():
    s = 64
    L = 4.0
    mode = 3
    x = torch.arange(s, dtype=torch.float64) * (L / s)
    z = torch.sin(2.0 * math.pi * mode * x / L).unsqueeze(0)
    ux, _, _ = spectral_derivatives_1d(z, domain_length=L)
    expected = (2.0 * math.pi * mode / L) * torch.cos(2.0 * math.pi * mode * x / L).unsqueeze(0)
    assert torch.allclose(ux, expected, atol=1e-7, rtol=1e-7)


def test_feature_cache_key_distinguishes_domain_metadata():
    common = {"dataset_hash": "data", "split_hash": "split"}
    obs = {"obs": "full", "J": 64}
    surrogate_l1 = {"reservoir": "static", "domain_length": 1.0, "effective_nx": 64, "dx": 1.0 / 64, "sub": 1, "sim_dtype": "float32"}
    surrogate_l2 = {"reservoir": "static", "domain_length": 2.0, "effective_nx": 64, "dx": 2.0 / 64, "sub": 1, "sim_dtype": "float32"}
    key1 = make_feature_cache_key(**common, surrogate_config=surrogate_l1, observation_config=obs)
    key2 = make_feature_cache_key(**common, surrogate_config=surrogate_l2, observation_config=obs)
    assert key1["surrogate_hash"] != key2["surrogate_hash"]
