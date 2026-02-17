"""Backprop-free Koopman-Reservoir operator learning for 2D Darcy flow."""

from __future__ import annotations

import argparse
import os
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from cli_utils import add_data_mode_args, add_split_args, validate_data_mode_args
from koopman_reservoir_utils import (
    FixedReservoirEncoder,
    ReservoirParams,
    build_basis_2d,
    estimate_spectral_radius,
    fit_basis_coefficients,
    fit_koopman,
    measure_with_basis,
    ridge_fit_linear_map,
    set_global_seed,
    stabilize_koopman,
    to_device,
)
from viz_utils import plot_2d_comparison, plot_error_histogram, rel_l2

try:
    from utilities3 import MatReader, UnitGaussianNormalizer
except Exception:  # pragma: no cover - fallback for environments without h5py
    MatReader = None  # type: ignore[assignment]

    class UnitGaussianNormalizer:  # type: ignore[no-redef]
        def __init__(self, x: torch.Tensor, eps: float = 1.0e-5):
            self.mean = torch.mean(x, 0)
            self.std = torch.std(x, 0)
            self.eps = eps

        def encode(self, x: torch.Tensor) -> torch.Tensor:
            return (x - self.mean) / (self.std + self.eps)

        def decode(self, x: torch.Tensor) -> torch.Tensor:
            return (x * (self.std + self.eps)) + self.mean

        def to(self, device: torch.device) -> "UnitGaussianNormalizer":
            self.mean = self.mean.to(device)
            self.std = self.std.to(device)
            return self


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Backprop-free Koopman-Reservoir Operator (2D Darcy)")
    add_data_mode_args(
        parser,
        default_data_mode="separate_files",
        default_data_file="data/piececonst_r421_N1024_smooth1.mat",
        default_train_file="data/piececonst_r421_N1024_smooth1.mat",
        default_test_file="data/piececonst_r421_N1024_smooth2.mat",
    )
    add_split_args(parser, default_train_split=0.8, default_seed=0)

    parser.add_argument("--ntrain", type=int, default=1000, help="Number of training samples.")
    parser.add_argument("--ntest", type=int, default=100, help="Number of test samples.")
    parser.add_argument("--r", type=int, default=5, help="Spatial downsampling rate.")
    parser.add_argument("--grid-size", type=int, default=421, help="Original grid size before downsampling.")
    parser.add_argument(
        "--normalize",
        choices=("none", "unit_gaussian"),
        default="unit_gaussian",
        help="Normalization mode for coeff/sol fields.",
    )

    parser.add_argument(
        "--measure-basis",
        choices=("fourier", "random_fourier", "legendre", "chebyshev", "rbf", "sensor"),
        default="random_fourier",
        help="Basis for measurement operator M.",
    )
    parser.add_argument("--measure-dim", type=int, default=256, help="Measurement dimension p.")
    parser.add_argument(
        "--decoder-basis",
        choices=("grid", "fourier", "legendre", "chebyshev", "rbf"),
        default="grid",
        help="Decoder basis for output reconstruction.",
    )
    parser.add_argument("--decoder-dim", type=int, default=0, help="Decoder basis size Q (ignored for grid).")
    parser.add_argument("--rbf-sigma", type=float, default=0.05, help="Sigma for RBF basis.")
    parser.add_argument(
        "--random-fourier-scale",
        type=float,
        default=4.0,
        help="Frequency scale for random Fourier basis.",
    )
    parser.add_argument("--basis-normalize", action="store_true", help="L2-normalize basis rows.")

    parser.add_argument("--reservoir-dim", type=int, default=512, help="Reservoir dimension m_res.")
    parser.add_argument("--washout", type=int, default=8, help="Reservoir iterations Lw.")
    parser.add_argument("--leak-alpha", type=float, default=1.0, help="Reservoir leak alpha in (0,1].")
    parser.add_argument("--spectral-radius", type=float, default=0.9, help="Target spectral radius for W.")
    parser.add_argument("--input-scale", type=float, default=1.0, help="Input matrix scale for reservoir U.")
    parser.add_argument("--bias-scale", type=float, default=0.0, help="Bias vector scale for reservoir b.")

    parser.add_argument("--ridge-k", type=float, default=1e-6, help="Ridge regularization for Koopman K.")
    parser.add_argument("--ridge-d", type=float, default=1e-6, help="Ridge regularization for decoder D.")
    parser.add_argument("--stabilize-k", action="store_true", help="Scale K so spectral radius <= 1.")
    parser.add_argument("--smoke-test", action="store_true", help="Run with synthetic data (no MAT files required).")
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Computation device.",
    )
    return parser


def _validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if not args.smoke_test:
        validate_data_mode_args(args, parser)
    if args.ntrain <= 0 or args.ntest <= 0:
        parser.error("--ntrain and --ntest must be positive.")
    if args.r <= 0:
        parser.error("--r must be positive.")
    if args.grid_size <= 1:
        parser.error("--grid-size must be > 1.")
    if args.measure_dim <= 0:
        parser.error("--measure-dim must be positive.")
    if args.decoder_basis != "grid" and args.decoder_dim <= 0:
        parser.error("--decoder-dim must be positive when --decoder-basis is not grid.")
    if args.reservoir_dim <= 0:
        parser.error("--reservoir-dim must be positive.")
    if args.washout <= 0:
        parser.error("--washout must be >= 1.")
    if not (0.0 < args.leak_alpha <= 1.0):
        parser.error("--leak-alpha must be in (0,1].")
    if args.spectral_radius <= 0:
        parser.error("--spectral-radius must be positive.")
    if args.ridge_k < 0 or args.ridge_d < 0:
        parser.error("--ridge-k and --ridge-d must be non-negative.")


def _split_train_test(
    x_data: torch.Tensor,
    y_data: torch.Tensor,
    ntrain: int,
    ntest: int,
    train_split: float,
    shuffle: bool,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    total = x_data.shape[0]
    idx = np.arange(total)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(idx)

    split_idx = int(total * train_split)
    train_idx = idx[:split_idx]
    test_idx = idx[split_idx:]
    if ntrain > len(train_idx) or ntest > len(test_idx):
        raise ValueError(
            f"Not enough samples: total={total}, train_split={train_split}, "
            f"available_train={len(train_idx)}, available_test={len(test_idx)}, "
            f"requested ntrain={ntrain}, ntest={ntest}."
        )
    train_idx = train_idx[:ntrain]
    test_idx = test_idx[:ntest]
    return x_data[train_idx], y_data[train_idx], x_data[test_idx], y_data[test_idx]


def _read_darcy_fields(path: str) -> tuple[torch.Tensor, torch.Tensor]:
    if MatReader is None:
        raise ImportError(
            "Failed to import utilities3.MatReader (likely missing h5py). "
            "Use --smoke-test or install h5py for MAT loading."
        )
    reader = MatReader(path)

    def _read_one(candidates: tuple[str, ...]) -> torch.Tensor:
        for key in candidates:
            try:
                return reader.read_field(key)
            except Exception:
                continue
        raise KeyError(f"None of fields {candidates} found in '{path}'.")

    coeff = _read_one(("coeff", "a"))
    sol = _read_one(("sol", "u"))
    return coeff.float(), sol.float()


def _downsample_2d(field: torch.Tensor, r: int, s: int) -> torch.Tensor:
    return field[:, ::r, ::r][:, :s, :s]


def _generate_smoke_data(ntrain: int, ntest: int, s: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    n = ntrain + ntest
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)

    coeff = torch.rand((n, s, s), generator=gen)
    coeff = F.avg_pool2d(coeff.unsqueeze(1), kernel_size=5, stride=1, padding=2).squeeze(1)

    x = torch.linspace(0.0, 1.0, s)
    y = torch.linspace(0.0, 1.0, s)
    xx, yy = torch.meshgrid(x, y, indexing="ij")
    pattern = torch.sin(2.0 * np.pi * xx) * torch.cos(2.0 * np.pi * yy)
    amp = 0.2 * torch.randn((n, 1, 1), generator=gen)
    coeff = coeff + amp * pattern.unsqueeze(0)

    smooth1 = F.avg_pool2d(coeff.unsqueeze(1), kernel_size=3, stride=1, padding=1).squeeze(1)
    smooth2 = F.avg_pool2d(coeff.unsqueeze(1), kernel_size=7, stride=1, padding=3).squeeze(1)
    sol = 0.65 * smooth1 + 0.35 * smooth2 + 0.08 * coeff * coeff
    return coeff.float(), sol.float()


def _load_data(args: argparse.Namespace) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    if args.smoke_test:
        s = 33
        coeff, sol = _generate_smoke_data(args.ntrain, args.ntest, s=s, seed=args.seed)
        x_train = coeff[: args.ntrain]
        y_train = sol[: args.ntrain]
        x_test = coeff[args.ntrain : args.ntrain + args.ntest]
        y_test = sol[args.ntrain : args.ntrain + args.ntest]
        return x_train, y_train, x_test, y_test, s

    s = int(((args.grid_size - 1) / args.r) + 1)
    if args.data_mode == "single_split":
        x_data, y_data = _read_darcy_fields(args.data_file)
        x_data = _downsample_2d(x_data, args.r, s)
        y_data = _downsample_2d(y_data, args.r, s)
        x_train, y_train, x_test, y_test = _split_train_test(
            x_data,
            y_data,
            ntrain=args.ntrain,
            ntest=args.ntest,
            train_split=args.train_split,
            shuffle=args.shuffle,
            seed=args.seed,
        )
        return x_train, y_train, x_test, y_test, s

    x_train_all, y_train_all = _read_darcy_fields(args.train_file)
    x_test_all, y_test_all = _read_darcy_fields(args.test_file)
    x_train_all = _downsample_2d(x_train_all, args.r, s)
    y_train_all = _downsample_2d(y_train_all, args.r, s)
    x_test_all = _downsample_2d(x_test_all, args.r, s)
    y_test_all = _downsample_2d(y_test_all, args.r, s)

    if args.ntrain > x_train_all.shape[0] or args.ntest > x_test_all.shape[0]:
        raise ValueError(
            f"Not enough samples in separate_files mode: train={x_train_all.shape[0]}, "
            f"test={x_test_all.shape[0]}, requested ntrain={args.ntrain}, ntest={args.ntest}."
        )

    return (
        x_train_all[: args.ntrain],
        y_train_all[: args.ntrain],
        x_test_all[: args.ntest],
        y_test_all[: args.ntest],
        s,
    )


def _fit_decoder(
    z_out_train: torch.Tensor,
    y_train_flat: torch.Tensor,
    x_grid: torch.Tensor,
    y_grid: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    if args.decoder_basis == "grid":
        coef = ridge_fit_linear_map(z_out_train, y_train_flat, ridge=args.ridge_d)  # [M, S]
        return coef, None

    phi = build_basis_2d(
        args.decoder_basis,
        dim=args.decoder_dim,
        x_grid=x_grid,
        y_grid=y_grid,
        normalize=args.basis_normalize,
        rbf_sigma=args.rbf_sigma,
        random_fourier_scale=args.random_fourier_scale,
        seed=args.seed + 131,
    )
    c_train = fit_basis_coefficients(y_train_flat, phi, ridge=args.ridge_d)  # [N, Q]
    coef = ridge_fit_linear_map(z_out_train, c_train, ridge=args.ridge_d)  # [M, Q]
    return coef, phi


def _decode(z: torch.Tensor, decoder_coef: torch.Tensor, decoder_phi: Optional[torch.Tensor]) -> torch.Tensor:
    if decoder_phi is None:
        return z @ decoder_coef
    c = z @ decoder_coef
    return c @ decoder_phi


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    _validate_args(args, parser)

    set_global_seed(args.seed)
    device = to_device(args.device)
    print(f"[info] device={device}")

    x_train_raw, y_train_raw, x_test_raw, y_test_raw, s = _load_data(args)
    print(
        f"[info] smoke_test={args.smoke_test}, train={x_train_raw.shape[0]}, "
        f"test={x_test_raw.shape[0]}, grid={s}x{s}"
    )

    x_normalizer: Optional[UnitGaussianNormalizer] = None
    y_normalizer: Optional[UnitGaussianNormalizer] = None

    if args.normalize == "unit_gaussian":
        x_normalizer = UnitGaussianNormalizer(x_train_raw)
        y_normalizer = UnitGaussianNormalizer(y_train_raw)
        x_train = x_normalizer.encode(x_train_raw)
        y_train = y_normalizer.encode(y_train_raw)
        x_test = x_normalizer.encode(x_test_raw)
        y_test = y_normalizer.encode(y_test_raw)
    else:
        x_train = x_train_raw
        y_train = y_train_raw
        x_test = x_test_raw
        y_test = y_test_raw

    x_train = x_train.to(device)
    y_train = y_train.to(device)
    x_test = x_test.to(device)
    y_test = y_test.to(device)

    ntrain = x_train.shape[0]
    ntest = x_test.shape[0]
    s_total = s * s

    x_train_flat = x_train.reshape(ntrain, s_total)
    y_train_flat = y_train.reshape(ntrain, s_total)
    x_test_flat = x_test.reshape(ntest, s_total)

    x_grid = torch.linspace(0.0, 1.0, s, dtype=torch.float32, device=device)
    y_grid = torch.linspace(0.0, 1.0, s, dtype=torch.float32, device=device)
    dxdy = float((1.0 / max(s - 1, 1)) ** 2)

    psi = build_basis_2d(
        args.measure_basis,
        dim=args.measure_dim,
        x_grid=x_grid,
        y_grid=y_grid,
        normalize=args.basis_normalize,
        rbf_sigma=args.rbf_sigma,
        random_fourier_scale=args.random_fourier_scale,
        seed=args.seed + 17,
    )

    m_in_train = measure_with_basis(x_train_flat, psi, dx=dxdy)
    m_out_train = measure_with_basis(y_train_flat, psi, dx=dxdy)
    m_in_test = measure_with_basis(x_test_flat, psi, dx=dxdy)

    reservoir = FixedReservoirEncoder(
        ReservoirParams(
            measure_dim=args.measure_dim,
            reservoir_dim=args.reservoir_dim,
            washout=args.washout,
            leak_alpha=args.leak_alpha,
            spectral_radius=args.spectral_radius,
            input_scale=args.input_scale,
            bias_scale=args.bias_scale,
            seed=args.seed,
            device=device,
        )
    )
    print(
        "[info] reservoir_W_radius: initial={:.4f}, scaled={:.4f}".format(
            reservoir.init_radius,
            reservoir.scaled_radius,
        )
    )

    z_in_train = reservoir.encode(m_in_train)
    z_out_train = reservoir.encode(m_out_train)
    z_in_test = reservoir.encode(m_in_test)

    kt = fit_koopman(z_in_train, z_out_train, ridge_k=args.ridge_k)
    if args.stabilize_k:
        kt, rho_before, rho_after = stabilize_koopman(kt, max_radius=1.0)
    else:
        rho_before = estimate_spectral_radius(kt)
        rho_after = rho_before
    print(f"[info] Koopman spectral radius: before={rho_before:.4f}, after={rho_after:.4f}")

    decoder_coef, decoder_phi = _fit_decoder(z_out_train, y_train_flat, x_grid, y_grid, args)

    z_out_test_hat = z_in_test @ kt
    y_test_hat_flat = _decode(z_out_test_hat, decoder_coef, decoder_phi)
    y_test_hat_norm = y_test_hat_flat.reshape(ntest, s, s)

    if y_normalizer is not None:
        y_normalizer = y_normalizer.to(device)
        pred = y_normalizer.decode(y_test_hat_norm)
        gt = y_test_raw.to(device)
    else:
        pred = y_test_hat_norm
        gt = y_test_raw.to(device)

    pred_cpu = pred.detach().cpu()
    gt_cpu = gt.detach().cpu()
    coeff_cpu = x_test_raw.detach().cpu()

    per_sample_err = [rel_l2(pred_cpu[i], gt_cpu[i]) for i in range(ntest)]
    err_mean = float(np.mean(per_sample_err))
    err_med = float(np.median(per_sample_err))
    print(f"[result] test relL2 mean={err_mean:.6f}, median={err_med:.6f}")

    viz_dir = os.path.join("visualizations", "koopman_reservoir_2d")
    os.makedirs(viz_dir, exist_ok=True)
    plot_error_histogram(per_sample_err, os.path.join(viz_dir, "test_relL2_hist"))

    for i in range(min(3, ntest)):
        plot_2d_comparison(
            gt=gt_cpu[i],
            pred=pred_cpu[i],
            input_field=coeff_cpu[i],
            out_path_no_ext=os.path.join(viz_dir, f"sample_{i:03d}"),
        )


if __name__ == "__main__":
    main()
