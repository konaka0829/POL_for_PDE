"""Backprop-free Koopman-Reservoir operator learning for 1D Burgers."""

from __future__ import annotations

import argparse
import os
from typing import Tuple

import numpy as np
import scipy.io
import torch

from cli_utils import add_data_mode_args, add_split_args, validate_data_mode_args
from koopman_reservoir_utils import (
    FixedReservoirEncoder,
    ReservoirParams,
    build_basis_1d,
    estimate_spectral_radius,
    fit_basis_coefficients,
    fit_koopman,
    measure_with_basis,
    ridge_fit_linear_map,
    set_global_seed,
    stabilize_koopman,
    to_device,
)
from viz_utils import plot_1d_prediction, plot_error_histogram, rel_l2


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Backprop-free Koopman-Reservoir Operator (1D Burgers)")
    add_data_mode_args(
        parser,
        default_data_mode="single_split",
        default_data_file="data/burgers_data_R10.mat",
        default_train_file=None,
        default_test_file=None,
    )
    add_split_args(parser, default_train_split=0.8, default_seed=0)

    parser.add_argument("--ntrain", type=int, default=1000, help="Number of training samples.")
    parser.add_argument("--ntest", type=int, default=100, help="Number of test samples.")
    parser.add_argument("--sub", type=int, default=2**3, help="Spatial subsampling rate.")

    parser.add_argument(
        "--measure-basis",
        choices=("fourier", "random_fourier", "legendre", "chebyshev", "rbf", "sensor"),
        default="fourier",
        help="Basis for measurement operator M.",
    )
    parser.add_argument("--measure-dim", type=int, default=128, help="Measurement dimension p.")
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
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Computation device.",
    )
    return parser


def _validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    validate_data_mode_args(args, parser)
    if args.ntrain <= 0 or args.ntest <= 0:
        parser.error("--ntrain and --ntest must be positive.")
    if args.sub <= 0:
        parser.error("--sub must be positive.")
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


def _to_timeseries(u: torch.Tensor, s_full: int) -> torch.Tensor:
    """Convert u to [N, T, S_full] if possible."""
    if u.ndim == 2:
        return u[:, None, :]
    if u.ndim != 3:
        raise ValueError(f"Expected u to be 2D or 3D, got shape={tuple(u.shape)}")
    if u.shape[2] == s_full:
        return u
    if u.shape[1] == s_full:
        return u.transpose(1, 2)
    raise ValueError(
        "For 3D u, expected shape [N,T,S] or [N,S,T] compatible with a.shape[1]. "
        f"Got u={tuple(u.shape)}, a_spatial={s_full}."
    )


def _build_pairs(a: torch.Tensor, u: torch.Tensor, sub: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build one-step pairs from Burgers data.

    Returns:
        x_pairs: [Npairs, S]
        y_pairs: [Npairs, S]
    """
    if a.ndim != 2:
        raise ValueError(f"Expected a to be [N,S], got {tuple(a.shape)}")
    s_full = a.shape[1]
    if sub > s_full:
        raise ValueError(f"Invalid sub={sub} for s_full={s_full}.")

    if u.ndim == 2:
        # Standard Burgers dataset: a -> u
        return a[:, ::sub], u[:, ::sub]

    u_ts = _to_timeseries(u, s_full=s_full)[:, :, ::sub]  # [N,T,S]
    if u_ts.shape[1] < 2:
        raise ValueError(f"Need at least 2 time steps in u for pair construction, got T={u_ts.shape[1]}.")
    s = u_ts.shape[2]
    x_pairs = u_ts[:, :-1, :].reshape(-1, s)
    y_pairs = u_ts[:, 1:, :].reshape(-1, s)
    return x_pairs, y_pairs


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


def _load_data(args: argparse.Namespace) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    def _read_mat_fields(path: str) -> tuple[torch.Tensor, torch.Tensor]:
        try:
            raw = scipy.io.loadmat(path)
        except Exception:
            try:
                import h5py  # type: ignore
            except ImportError as exc:
                raise ImportError(
                    "Failed to load MAT file with scipy.io.loadmat, and h5py is not installed "
                    "for v7.3 MAT fallback."
                ) from exc
            with h5py.File(path, "r") as f:
                if "a" not in f or "u" not in f:
                    raise KeyError(f"MAT file must contain 'a' and 'u': {path}")
                a_np = np.array(f["a"], dtype=np.float32).transpose()
                u_np = np.array(f["u"], dtype=np.float32).transpose()
        else:
            if "a" not in raw or "u" not in raw:
                raise KeyError(f"MAT file must contain 'a' and 'u': {path}")
            a_np = raw["a"].astype(np.float32)
            u_np = raw["u"].astype(np.float32)
        return torch.from_numpy(a_np), torch.from_numpy(u_np)

    if args.data_mode == "single_split":
        a, u = _read_mat_fields(args.data_file)
        x_data, y_data = _build_pairs(a, u, sub=args.sub)
        return _split_train_test(
            x_data,
            y_data,
            ntrain=args.ntrain,
            ntest=args.ntest,
            train_split=args.train_split,
            shuffle=args.shuffle,
            seed=args.seed,
        )

    a_train, u_train = _read_mat_fields(args.train_file)
    a_test, u_test = _read_mat_fields(args.test_file)

    x_train_all, y_train_all = _build_pairs(a_train, u_train, sub=args.sub)
    x_test_all, y_test_all = _build_pairs(a_test, u_test, sub=args.sub)
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
    )


def _fit_decoder(
    z0_train: torch.Tensor,
    z1_train: torch.Tensor,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_grid: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    # Decoder is trained on both (z0 -> u0) and (z1 -> u1) pairs for stability.
    z_dec = torch.cat([z0_train, z1_train], dim=0)  # [2N, M]
    u_dec = torch.cat([x_train, y_train], dim=0)  # [2N, S]

    if args.decoder_basis == "grid":
        coef = ridge_fit_linear_map(z_dec, u_dec, ridge=args.ridge_d)  # [M, S]
        return coef, None

    phi = build_basis_1d(
        args.decoder_basis,
        dim=args.decoder_dim,
        x_grid=x_grid,
        normalize=args.basis_normalize,
        rbf_sigma=args.rbf_sigma,
        random_fourier_scale=args.random_fourier_scale,
        seed=args.seed + 131,
    )
    c_dec = fit_basis_coefficients(u_dec, phi, ridge=args.ridge_d)  # [2N, Q]
    coef = ridge_fit_linear_map(z_dec, c_dec, ridge=args.ridge_d)  # [M, Q]
    return coef, phi


def _decode(z: torch.Tensor, decoder_coef: torch.Tensor, decoder_phi: torch.Tensor | None) -> torch.Tensor:
    if decoder_phi is None:
        # grid decoder: [N,M] @ [M,S] -> [N,S]
        return z @ decoder_coef
    # basis decoder: z -> c -> u
    c = z @ decoder_coef  # [N,Q]
    return c @ decoder_phi  # [N,S]


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    _validate_args(args, parser)

    set_global_seed(args.seed)
    device = to_device(args.device)
    print(f"[info] device={device}")

    x_train, y_train, x_test, y_test = _load_data(args)
    x_train = x_train.float().to(device)
    y_train = y_train.float().to(device)
    x_test = x_test.float().to(device)
    y_test = y_test.float().to(device)

    ntrain, s = x_train.shape
    ntest = x_test.shape[0]
    print(f"[info] train={ntrain}, test={ntest}, spatial_points={s}")

    x_grid = torch.linspace(0.0, 1.0, s, dtype=torch.float32, device=device)
    dx = float(1.0 / max(s - 1, 1))

    psi = build_basis_1d(
        args.measure_basis,
        dim=args.measure_dim,
        x_grid=x_grid,
        normalize=args.basis_normalize,
        rbf_sigma=args.rbf_sigma,
        random_fourier_scale=args.random_fourier_scale,
        seed=args.seed + 17,
    )

    # Measurements m0,m1: [N, P]
    m0_train = measure_with_basis(x_train, psi, dx=dx)
    m1_train = measure_with_basis(y_train, psi, dx=dx)
    m0_test = measure_with_basis(x_test, psi, dx=dx)

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

    # Latent states z0,z1: [N, M]
    z0_train = reservoir.encode(m0_train)
    z1_train = reservoir.encode(m1_train)
    z0_test = reservoir.encode(m0_test)

    # Koopman: z1 ≈ z0 @ Kt
    kt = fit_koopman(z0_train, z1_train, ridge_k=args.ridge_k)
    if args.stabilize_k:
        kt, rho_before, rho_after = stabilize_koopman(kt, max_radius=1.0)
    else:
        rho_before = estimate_spectral_radius(kt)
        rho_after = rho_before
    print(f"[info] Koopman spectral radius: before={rho_before:.4f}, after={rho_after:.4f}")

    decoder_coef, decoder_phi = _fit_decoder(z0_train, z1_train, x_train, y_train, x_grid, args)

    # One-step prediction on test: u_hat1 = Decode(K @ Encode(M(u0)))
    z1_test_hat = z0_test @ kt
    pred_test = _decode(z1_test_hat, decoder_coef, decoder_phi)

    # Metrics
    pred_cpu = pred_test.detach().cpu()
    gt_cpu = y_test.detach().cpu()
    x0_cpu = x_test.detach().cpu()
    per_sample_err = [rel_l2(pred_cpu[i], gt_cpu[i]) for i in range(ntest)]
    err_mean = float(np.mean(per_sample_err))
    err_med = float(np.median(per_sample_err))
    print(f"[result] test relL2 mean={err_mean:.6f}, median={err_med:.6f}")

    # Visualization (png/pdf/svg)
    viz_dir = os.path.join("visualizations", "koopman_reservoir_1d")
    os.makedirs(viz_dir, exist_ok=True)
    plot_error_histogram(per_sample_err, os.path.join(viz_dir, "test_relL2_hist"))

    sample_ids = [0, min(1, ntest - 1), min(2, ntest - 1)]
    x_np = np.linspace(0.0, 1.0, s)
    for i in sample_ids:
        plot_1d_prediction(
            x=x_np,
            gt=gt_cpu[i],
            pred=pred_cpu[i],
            input_u0=x0_cpu[i],
            out_path_no_ext=os.path.join(viz_dir, f"sample_{i:03d}"),
            title_prefix=f"sample {i}: ",
        )


if __name__ == "__main__":
    main()
