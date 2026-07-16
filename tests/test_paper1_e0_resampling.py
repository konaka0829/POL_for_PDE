import torch

from pol.paper1.grids import periodic_grid, spectral_resample_periodic


def test_odd_even_single_modes_and_nyquist():
    for n_in, n_out in ((15, 16), (16, 15), (16, 24), (15, 25)):
        x = periodic_grid(n_in, 1.0)
        y = periodic_grid(n_out, 1.0)
        for k in (1, 3):
            assert torch.allclose(spectral_resample_periodic(torch.cos(2 * torch.pi * k * x), n_out, domain_length=1.0), torch.cos(2 * torch.pi * k * y), atol=1e-12)
            assert torch.allclose(spectral_resample_periodic(torch.sin(2 * torch.pi * k * x), n_out, domain_length=1.0), torch.sin(2 * torch.pi * k * y), atol=1e-12)
    x = periodic_grid(16, 1.0); y = periodic_grid(32, 1.0)
    assert torch.allclose(spectral_resample_periodic(torch.cos(2 * torch.pi * 8 * x), 32, domain_length=1.0), torch.cos(2 * torch.pi * 8 * y), atol=1e-12)


def test_alias_prevention_counterexample():
    x64 = periodic_grid(64, 1.0); high = torch.cos(2 * torch.pi * 17 * x64)
    x16 = periodic_grid(16, 1.0)
    assert torch.allclose(high[::4], torch.cos(2 * torch.pi * x16), atol=1e-12)
    assert torch.max(torch.abs(spectral_resample_periodic(high, 16, domain_length=1.0))) < 1e-12


def test_independent_mode_dictionary_oracle_and_clone():
    x = periodic_grid(12, 1.0)
    values = 0.5 + 0.7 * torch.cos(2 * torch.pi * 2 * x) - 0.2 * torch.sin(2 * torch.pi * 3 * x)
    y = periodic_grid(19, 1.0)
    expected = 0.5 + 0.7 * torch.cos(2 * torch.pi * 2 * y) - 0.2 * torch.sin(2 * torch.pi * 3 * y)
    assert torch.allclose(spectral_resample_periodic(values, 19, domain_length=1.0), expected, atol=1e-12)
    clone = spectral_resample_periodic(values, 12, domain_length=1.0)
    assert clone.data_ptr() != values.data_ptr()
