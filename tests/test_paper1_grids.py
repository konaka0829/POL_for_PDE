import torch

from pol.paper1.grids import periodic_grid, spectral_resample_periodic


def _wave(nx, k, kind="cos", dtype=torch.float64):
    x = periodic_grid(nx, 1.0, dtype=dtype)
    phase = 2.0 * torch.pi * k * x
    return torch.cos(phase) if kind == "cos" else torch.sin(phase)


def test_periodic_grid_endpoint_free():
    x = periodic_grid(8, 2.0, dtype=torch.float64)
    assert x.shape == (8,)
    assert x[0].item() == 0.0
    assert x[-1].item() == 1.75
    assert torch.allclose(torch.diff(x), torch.full((7,), 0.25, dtype=torch.float64))


def test_constant_resampling():
    u = 3.0 * torch.ones(2, 7, dtype=torch.float64)
    got = spectral_resample_periodic(u, 12, domain_length=1.0)
    assert got.shape == (2, 12)
    assert torch.allclose(got, 3.0 * torch.ones_like(got), atol=1e-12)


def test_sin_cos_shared_modes_even_odd():
    for n_in, n_out in [(16, 15), (15, 16), (16, 24)]:
        u = _wave(n_in, 3, "cos") + 0.25 * _wave(n_in, 2, "sin")
        got = spectral_resample_periodic(u, n_out, domain_length=1.0)
        expected = _wave(n_out, 3, "cos") + 0.25 * _wave(n_out, 2, "sin")
        assert torch.allclose(got, expected, atol=1e-10, rtol=1e-10)


def test_even_grid_nyquist_cosine_upsample():
    n_in, n_out = 8, 18
    u = torch.cos(torch.pi * torch.arange(n_in, dtype=torch.float64))
    got = spectral_resample_periodic(u, n_out, domain_length=1.0)
    x = periodic_grid(n_out, 1.0)
    expected = torch.cos(2.0 * torch.pi * (n_in // 2) * x)
    assert torch.allclose(got, expected, atol=1e-10)


def test_low_band_round_trip():
    u = _wave(32, 4, "cos") - 0.7 * _wave(32, 5, "sin")
    down = spectral_resample_periodic(u, 17, domain_length=1.0)
    up = spectral_resample_periodic(down, 32, domain_length=1.0)
    assert torch.allclose(up, u, atol=1e-10, rtol=1e-10)


def test_batch_dtype_and_identity():
    u32 = torch.stack([_wave(10, 2, dtype=torch.float32), 2.0 * _wave(10, 1, "sin", dtype=torch.float32)])
    out = spectral_resample_periodic(u32, 14, domain_length=1.0)
    assert out.shape == (2, 14)
    assert out.dtype == torch.float32
    same = spectral_resample_periodic(u32, 10, domain_length=1.0)
    assert torch.allclose(same, u32)
    assert same.data_ptr() != u32.data_ptr()
