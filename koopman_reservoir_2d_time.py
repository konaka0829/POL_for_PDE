"""Backprop-free Koopman-Reservoir operator learning for 2D time-series rollout."""

from __future__ import annotations

import argparse
import os
from typing import Optional

import numpy as np
import torch

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
from utilities3 import MatReader
from viz_utils import (
    plot_2d_time_slices,
    plot_error_histogram,
    plot_rel_l2_over_time,
    rel_l2,
)


class TimeGaussianNormalizer2D:
    """Time-independent per-pixel normalization for 2D trajectories [N,S,S,T]."""

    def __init__(self, u_train: torch.Tensor, eps: float = 1e-5):
        self.eps = eps
        self.mean = torch.mean(u_train, dim=(0, 3), keepdim=True)
        self.std = torch.std(u_train, dim=(0, 3), keepdim=True)

    def encode(self, u: torch.Tensor) -> torch.Tensor:
        return (u - self.mean) / (self.std + self.eps)

    def decode(self, u: torch.Tensor) -> torch.Tensor:
        return u * (self.std + self.eps) + self.mean

    def to(self, device: torch.device) -> "TimeGaussianNormalizer2D":
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        return self


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Backprop-free Koopman-Reservoir Operator (2D time-series)")
    add_data_mode_args(
        p,
        default_data_mode="single_split",
        default_data_file="data/allen_cahn_2d_ts.mat",
        default_train_file=None,
        default_test_file=None,
    )
    add_split_args(p, default_train_split=0.8, default_seed=0)

    p.add_argument("--field", type=str, default="u", help="MAT field name")
    p.add_argument("--ntrain", type=int, default=100, help="Number of train trajectories")
    p.add_argument("--ntest", type=int, default=20, help="Number of test trajectories")
    p.add_argument("--sub", type=int, default=1, help="Spatial subsampling")
    p.add_argument("--S", type=int, default=64, help="Expected spatial size after subsampling")
    p.add_argument("--T", type=int, default=40, help="Rollout horizon")
    p.add_argument("--t0", type=int, default=0, help="Rollout start time index")
    p.add_argument("--normalize", choices=("none", "unit_gaussian"), default="none")

    p.add_argument(
        "--measure-basis",
        choices=("fourier", "random_fourier", "legendre", "chebyshev", "rbf", "sensor"),
        default="random_fourier",
    )
    p.add_argument("--measure-dim", type=int, default=256)
    p.add_argument(
        "--decoder-basis",
        choices=("grid", "fourier", "legendre", "chebyshev", "rbf"),
        default="grid",
    )
    p.add_argument("--decoder-dim", type=int, default=0)
    p.add_argument("--rbf-sigma", type=float, default=0.05)
    p.add_argument("--random-fourier-scale", type=float, default=4.0)
    p.add_argument("--basis-normalize", action="store_true")

    p.add_argument("--reservoir-dim", type=int, default=512)
    p.add_argument("--washout", type=int, default=8)
    p.add_argument("--leak-alpha", type=float, default=1.0)
    p.add_argument("--spectral-radius", type=float, default=0.9)
    p.add_argument("--input-scale", type=float, default=1.0)
    p.add_argument("--bias-scale", type=float, default=0.0)

    p.add_argument("--ridge-k", type=float, default=1e-6)
    p.add_argument("--ridge-d", type=float, default=1e-6)
    p.add_argument("--stabilize-k", action="store_true")
    p.add_argument("--smoke-test", action="store_true")
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return p


def _validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if not args.smoke_test:
        validate_data_mode_args(args, parser)
    if args.ntrain <= 0 or args.ntest <= 0:
        parser.error("--ntrain and --ntest must be positive")
    if args.sub <= 0 or args.S <= 1 or args.T <= 0:
        parser.error("--sub, --S and --T must be positive")
    if args.measure_dim <= 0:
        parser.error("--measure-dim must be positive")
    if args.decoder_basis != "grid" and args.decoder_dim <= 0:
        parser.error("--decoder-dim must be positive for non-grid decoder")


def _to_time_last_2d(u: torch.Tensor) -> torch.Tensor:
    if u.ndim != 4:
        raise ValueError(f"Expected 4D u, got {tuple(u.shape)}")
    # If already [N,S,S,T], keep. Otherwise swap last two if [N,T,S,S].
    if u.shape[1] == u.shape[2]:
        return u
    if u.shape[2] == u.shape[3]:
        return u.permute(0, 2, 3, 1)
    raise ValueError(f"Cannot infer time-last format from shape={tuple(u.shape)}")


def _generate_smoke_u(n: int, s: int, t_total: int, seed: int) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    k = 2.0 * np.pi * np.fft.fftfreq(s, d=1.0 / s)
    kx, ky = np.meshgrid(k, k, indexing="ij")
    k2 = kx * kx + ky * ky

    out = np.zeros((n, s, s, t_total), dtype=np.float32)
    for i in range(n):
        u0 = rng.normal(0.0, 1.0, size=(s, s)).astype(np.float64)
        u_hat0 = np.fft.fft2(u0)
        u_hat0 *= np.exp(-0.02 * k2)
        for t in range(t_total):
            decay = np.exp(-0.02 * k2 * t)
            ut = np.fft.ifft2(u_hat0 * decay).real
            out[i, :, :, t] = ut.astype(np.float32)
    return torch.from_numpy(out)


def _read_u(path: str, field: str, sub: int) -> torch.Tensor:
    u = MatReader(path).read_field(field).float()
    u = _to_time_last_2d(u)
    return u[:, ::sub, ::sub, :]


def _split_traj_indices(total: int, train_split: float, shuffle: bool, seed: int) -> tuple[np.ndarray, np.ndarray]:
    idx = np.arange(total)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(idx)
    split = int(total * train_split)
    return idx[:split], idx[split:]


def _fit_decoder(
    z_train_all: torch.Tensor,
    u_train_all: torch.Tensor,
    x_grid: torch.Tensor,
    y_grid: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    if args.decoder_basis == "grid":
        coef = ridge_fit_linear_map(z_train_all, u_train_all, ridge=args.ridge_d)
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
    c = fit_basis_coefficients(u_train_all, phi, ridge=args.ridge_d)
    coef = ridge_fit_linear_map(z_train_all, c, ridge=args.ridge_d)
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

    if args.smoke_test:
        t_total = max(args.T + args.t0 + 1, 50)
        u_all = _generate_smoke_u(args.ntrain + args.ntest, s=args.S, t_total=t_total, seed=args.seed)
        u_train_raw = u_all[: args.ntrain]
        u_test_raw = u_all[args.ntrain : args.ntrain + args.ntest]
    elif args.data_mode == "single_split":
        u_raw = _read_u(args.data_file, args.field, args.sub)
        train_idx, test_idx = _split_traj_indices(u_raw.shape[0], args.train_split, args.shuffle, args.seed)
        if args.ntrain > len(train_idx) or args.ntest > len(test_idx):
            raise ValueError(
                f"Not enough trajectories: total={u_raw.shape[0]}, available_train={len(train_idx)}, "
                f"available_test={len(test_idx)}, requested ntrain={args.ntrain}, ntest={args.ntest}"
            )
        u_train_raw = u_raw[train_idx[: args.ntrain]]
        u_test_raw = u_raw[test_idx[: args.ntest]]
    else:
        u_train_raw = _read_u(args.train_file, args.field, args.sub)
        u_test_raw = _read_u(args.test_file, args.field, args.sub)
        if args.ntrain > u_train_raw.shape[0] or args.ntest > u_test_raw.shape[0]:
            raise ValueError(
                f"Not enough trajectories in separate_files mode: train={u_train_raw.shape[0]}, test={u_test_raw.shape[0]}, "
                f"requested ntrain={args.ntrain}, ntest={args.ntest}"
            )
        u_train_raw = u_train_raw[: args.ntrain]
        u_test_raw = u_test_raw[: args.ntest]

    ntrain, s1, s2, t_total = u_train_raw.shape
    if s1 != s2:
        raise ValueError(f"Expected square spatial grid, got {s1}x{s2}")
    if args.S != s1:
        raise ValueError(f"Expected S={args.S} after subsampling, got {s1}")
    ntest = u_test_raw.shape[0]

    t_eval = min(args.T, t_total - args.t0 - 1)
    if t_eval <= 0:
        raise ValueError(f"Invalid t0/T for T_total={t_total}: t0={args.t0}, T={args.T}")

    normalizer: Optional[TimeGaussianNormalizer2D] = None
    if args.normalize == "unit_gaussian":
        normalizer = TimeGaussianNormalizer2D(u_train_raw)
        u_train = normalizer.encode(u_train_raw)
        u_test = normalizer.encode(u_test_raw)
    else:
        u_train = u_train_raw
        u_test = u_test_raw

    u_train = u_train.to(device)
    u_test = u_test.to(device)

    s_total = args.S * args.S
    x_pairs = u_train[..., :-1].permute(0, 3, 1, 2).reshape(-1, s_total)
    y_pairs = u_train[..., 1:].permute(0, 3, 1, 2).reshape(-1, s_total)

    x_grid = torch.linspace(0.0, 1.0, args.S, dtype=torch.float32, device=device)
    y_grid = torch.linspace(0.0, 1.0, args.S, dtype=torch.float32, device=device)
    dxdy = float((1.0 / max(args.S - 1, 1)) ** 2)

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

    m0 = measure_with_basis(x_pairs, psi, dx=dxdy)
    m1 = measure_with_basis(y_pairs, psi, dx=dxdy)

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

    z0 = reservoir.encode(m0)
    z1 = reservoir.encode(m1)
    kt = fit_koopman(z0, z1, ridge_k=args.ridge_k)
    if args.stabilize_k:
        kt, rho_before, rho_after = stabilize_koopman(kt, max_radius=1.0)
    else:
        rho_before = estimate_spectral_radius(kt)
        rho_after = rho_before
    print(f"[info] Koopman spectral radius: before={rho_before:.4f}, after={rho_after:.4f}")

    # Decoder fit on all train times.
    u_train_all = u_train.permute(0, 3, 1, 2).reshape(-1, s_total)
    m_train_all = measure_with_basis(u_train_all, psi, dx=dxdy)
    z_train_all = reservoir.encode(m_train_all)
    decoder_coef, decoder_phi = _fit_decoder(z_train_all, u_train_all, x_grid, y_grid, args)

    pred_all = torch.zeros((ntest, args.S, args.S, t_eval), dtype=torch.float32, device=device)

    with torch.no_grad():
        for i in range(ntest):
            u0 = u_test[i, :, :, args.t0].reshape(1, s_total)
            m = measure_with_basis(u0, psi, dx=dxdy)
            z = reservoir.encode(m)
            for k in range(t_eval):
                z = z @ kt
                u_hat = _decode(z, decoder_coef, decoder_phi)
                pred_all[i, :, :, k] = u_hat.reshape(args.S, args.S)

    gt_all = u_test[:, :, :, args.t0 + 1 : args.t0 + 1 + t_eval]

    if normalizer is not None:
        normalizer = normalizer.to(device)
        pred_eval = normalizer.decode(pred_all)
        gt_eval = u_test_raw[:, :, :, args.t0 + 1 : args.t0 + 1 + t_eval].to(device)
    else:
        pred_eval = pred_all
        gt_eval = gt_all

    pred_cpu = pred_eval.detach().cpu()
    gt_cpu = gt_eval.detach().cpu()

    full_err = []
    per_step_err = np.zeros((ntest, t_eval), dtype=np.float64)
    for i in range(ntest):
        full_err.append(rel_l2(pred_cpu[i], gt_cpu[i]))
        for t in range(t_eval):
            per_step_err[i, t] = rel_l2(pred_cpu[i, :, :, t], gt_cpu[i, :, :, t])

    mean_full = float(np.mean(full_err))
    med_full = float(np.median(full_err))
    mean_curve = np.mean(per_step_err, axis=0)
    print(f"[result] test full relL2 mean={mean_full:.6f}, median={med_full:.6f}")
    print(f"[result] test step relL2 mean_over_time={float(np.mean(mean_curve)):.6f}")

    viz_dir = os.path.join("visualizations", "koopman_reservoir_2d_time")
    os.makedirs(viz_dir, exist_ok=True)

    plot_error_histogram(full_err, os.path.join(viz_dir, "test_relL2_hist"))

    # Plot relL2-over-time for a representative sample and time-slices for a few samples.
    sample_ids = [0, min(1, ntest - 1), min(2, ntest - 1)]
    t_ids = [0, t_eval // 2, t_eval - 1]
    for i in sample_ids:
        plot_2d_time_slices(
            gt=gt_cpu[i],
            pred=pred_cpu[i],
            t_indices=t_ids,
            out_path_no_ext=os.path.join(viz_dir, f"sample_{i:03d}_slices"),
        )
        plot_rel_l2_over_time(
            gt=gt_cpu[i],
            pred=pred_cpu[i],
            out_path_no_ext=os.path.join(viz_dir, f"sample_{i:03d}_relL2_over_time"),
        )


if __name__ == "__main__":
    main()
