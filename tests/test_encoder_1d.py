import argparse

import torch

from pol.encoder_1d import FixedEncoder1D


def make_args(**kwargs):
    base = dict(
        encoder="linear",
        encoder_center=0,
        encoder_standardize=0,
        encoder_standardize_eps=1e-6,
        input_scale=1.0,
        input_shift=0.0,
        encoder_post="none",
        encoder_tanh_gamma=1.0,
        encoder_clip_c=3.0,
        encoder_fourier_mode="lowpass",
        encoder_fourier_kmin=0,
        encoder_fourier_kmax=16,
        encoder_fourier_seed=0,
        encoder_fourier_amp_std=0.5,
        encoder_fourier_output_scale=1.0,
        encoder_randconv_kernel_size=33,
        encoder_randconv_seed=0,
        encoder_randconv_std=1.0,
        encoder_randconv_normalize="l2",
        encoder_fourier_rfm_C=1,
        encoder_fourier_rfm_mode="sum",
        encoder_fourier_rfm_activation="tanh",
        encoder_fourier_rfm_kmin=0,
        encoder_fourier_rfm_kmax=16,
        encoder_fourier_rfm_seed=0,
        encoder_fourier_rfm_theta_scale=1.0,
        encoder_fourier_rfm_output_scale=1.0,
        encoder_poly_a1=1.0,
        encoder_poly_a2=0.0,
        encoder_poly_a3=0.0,
    )
    base.update(kwargs)
    return argparse.Namespace(**base)


def test_center_standardize_properties():
    s = 64
    x = torch.randn(5, s)
    args = make_args(encoder="linear", encoder_center=1, encoder_standardize=1)
    enc = FixedEncoder1D(s=s, device=torch.device("cpu"), dtype=torch.float32, args=args)
    out = enc.encode(x)

    z = out.z0_list[0]
    mean = z.mean(dim=-1)
    std = z.std(dim=-1)
    assert torch.max(mean.abs()).item() < 1e-5
    assert torch.allclose(std, torch.ones_like(std), atol=1e-4, rtol=1e-3)


def test_tanh_and_clip_boundedness():
    s = 64
    x = 10.0 * torch.randn(4, s)

    args_tanh = make_args(encoder_post="tanh", encoder_tanh_gamma=1.7)
    enc_tanh = FixedEncoder1D(s=s, device=torch.device("cpu"), dtype=torch.float32, args=args_tanh)
    z_tanh = enc_tanh.encode(x).z0_list[0]
    assert torch.max(z_tanh.abs()).item() <= 1.0 + 1e-6

    c = 0.7
    args_clip = make_args(encoder_post="clip", encoder_clip_c=c)
    enc_clip = FixedEncoder1D(s=s, device=torch.device("cpu"), dtype=torch.float32, args=args_clip)
    z_clip = enc_clip.encode(x).z0_list[0]
    assert torch.max(z_clip).item() <= c + 1e-6
    assert torch.min(z_clip).item() >= -c - 1e-6


def test_fourier_filter_deterministic():
    s = 128
    x = torch.randn(3, s)
    args = make_args(
        encoder="fourier_filter",
        encoder_fourier_mode="randphase",
        encoder_fourier_kmin=2,
        encoder_fourier_kmax=16,
        encoder_fourier_seed=17,
    )
    e1 = FixedEncoder1D(s=s, device=torch.device("cpu"), dtype=torch.float32, args=args)
    e2 = FixedEncoder1D(s=s, device=torch.device("cpu"), dtype=torch.float32, args=args)
    y1 = e1.encode(x).z0_list[0]
    y2 = e2.encode(x).z0_list[0]
    assert torch.allclose(y1, y2)


def test_randconv_deterministic_and_shape():
    s = 128
    x = torch.randn(2, s)
    args = make_args(
        encoder="randconv",
        encoder_randconv_kernel_size=31,
        encoder_randconv_seed=3,
        encoder_randconv_std=0.8,
        encoder_randconv_normalize="l2",
    )
    e1 = FixedEncoder1D(s=s, device=torch.device("cpu"), dtype=torch.float32, args=args)
    e2 = FixedEncoder1D(s=s, device=torch.device("cpu"), dtype=torch.float32, args=args)
    y1 = e1.encode(x).z0_list[0]
    y2 = e2.encode(x).z0_list[0]
    assert y1.shape == (2, s)
    assert torch.allclose(y1, y2)


def test_poly_deriv_constant_derivative_is_zero():
    s = 64
    x = torch.full((3, s), 2.5)
    args = make_args(encoder="poly_deriv", encoder_poly_a1=0.0, encoder_poly_a2=0.0, encoder_poly_a3=1.0)
    enc = FixedEncoder1D(s=s, device=torch.device("cpu"), dtype=torch.float32, args=args)
    y = enc.encode(x).z0_list[0]
    assert torch.allclose(y, torch.zeros_like(y), atol=1e-5, rtol=1e-5)


def test_fourier_rfm_sum_and_ensemble_shapes():
    s = 64
    x = torch.randn(2, s)

    args_sum = make_args(
        encoder="fourier_rfm",
        encoder_fourier_rfm_C=4,
        encoder_fourier_rfm_mode="sum",
        encoder_fourier_rfm_seed=11,
    )
    enc_sum = FixedEncoder1D(s=s, device=torch.device("cpu"), dtype=torch.float32, args=args_sum)
    out_sum = enc_sum.encode(x)
    assert len(out_sum.z0_list) == 1

    args_ens = make_args(
        encoder="fourier_rfm",
        encoder_fourier_rfm_C=4,
        encoder_fourier_rfm_mode="ensemble",
        encoder_fourier_rfm_seed=11,
    )
    enc_ens = FixedEncoder1D(s=s, device=torch.device("cpu"), dtype=torch.float32, args=args_ens)
    out_ens = enc_ens.encode(x)
    assert len(out_ens.z0_list) == 4
    for z in out_ens.z0_list:
        assert z.shape == (2, s)
