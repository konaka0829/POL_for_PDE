#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from dataclasses import asdict, dataclass

import numpy as np
import torch

from pde_features import (
    PDERandomFeatureMap1D,
    apply_advection_1d,
    apply_convection_diffusion_1d,
    apply_heat_semigroup_1d,
    apply_wave_1d,
    set_random_seed,
    to_torch_dtype,
)
from ridge import solve_ridge
from utilities3 import LpLoss, MatReader
from viz_utils import plot_1d_prediction, plot_error_histogram, rel_l2


@dataclass
class RunMetrics:
    train_rel_l2: float
    test_rel_l2: float
    ntrain: int
    ntest: int
    s: int


def _choose_device(solve_device: str) -> torch.device:
    solve_device = solve_device.lower()
    if solve_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if solve_device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--solve-device cuda requested but CUDA is not available")
        return torch.device("cuda")
    if solve_device == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Unsupported solve-device: {solve_device}")


def _compute_feature_matrix(
    feature_map: PDERandomFeatureMap1D,
    x: torch.Tensor,
    batch_size: int,
) -> torch.Tensor:
    rows = []
    with torch.no_grad():
        for start in range(0, x.shape[0], batch_size):
            end = min(start + batch_size, x.shape[0])
            rows.append(feature_map.features(x[start:end]))
    return torch.cat(rows, dim=0)


def _load_data(args: argparse.Namespace, dtype: torch.dtype):
    if args.smoke_test:
        s = 128
        ntrain = min(args.ntrain, 64)
        ntest = min(args.ntest, 16)
        total = ntrain + ntest

        a = torch.randn(total, s, dtype=dtype)
        tau_true = 0.08
        nu_true = args.nu
        c_true = 0.7
        c_wave_true = 1.25
        gamma_true = max(args.wave_gamma, 0.2)

        if args.operator == "heat":
            u = apply_heat_semigroup_1d(a, tau=tau_true, nu=nu_true)
        elif args.operator == "advection":
            u = apply_advection_1d(a, tau=tau_true, c=c_true)
        elif args.operator == "convdiff":
            u = apply_convection_diffusion_1d(a, tau=tau_true, nu=nu_true, c=c_true)
        else:
            u = apply_wave_1d(a, tau=tau_true, c_wave=c_wave_true, gamma=gamma_true)

        x_train = a[:ntrain].unsqueeze(-1)
        y_train = u[:ntrain]
        x_test = a[ntrain : ntrain + ntest].unsqueeze(-1)
        y_test = u[ntrain : ntrain + ntest]
        return x_train, y_train, x_test, y_test

    if args.data_mode == "single_split":
        if not args.data_file:
            raise ValueError("--data-file is required for --data-mode single_split")
        reader = MatReader(args.data_file)
        x_data = reader.read_field("a")[:, :: args.sub].to(dtype=dtype)
        y_data = reader.read_field("u")[:, :: args.sub].to(dtype=dtype)

        n_total = x_data.shape[0]
        if args.shuffle:
            g = torch.Generator(device="cpu")
            g.manual_seed(args.seed)
            perm = torch.randperm(n_total, generator=g)
            x_data = x_data[perm]
            y_data = y_data[perm]

        split_idx = int(n_total * args.train_split)
        split_idx = max(1, min(split_idx, n_total - 1))
        ntrain = min(args.ntrain, split_idx)
        ntest = min(args.ntest, n_total - split_idx)
        if ntest <= 0:
            raise ValueError("No test data available with the current train/test split")

        x_train = x_data[:ntrain]
        y_train = y_data[:ntrain]
        x_test = x_data[split_idx : split_idx + ntest]
        y_test = y_data[split_idx : split_idx + ntest]

    else:
        if not args.train_file or not args.test_file:
            raise ValueError("--train-file and --test-file are required for --data-mode separate_files")
        train_reader = MatReader(args.train_file)
        test_reader = MatReader(args.test_file)

        x_train_all = train_reader.read_field("a")[:, :: args.sub].to(dtype=dtype)
        y_train_all = train_reader.read_field("u")[:, :: args.sub].to(dtype=dtype)
        x_test_all = test_reader.read_field("a")[:, :: args.sub].to(dtype=dtype)
        y_test_all = test_reader.read_field("u")[:, :: args.sub].to(dtype=dtype)

        if args.shuffle:
            g = torch.Generator(device="cpu")
            g.manual_seed(args.seed)
            perm = torch.randperm(x_train_all.shape[0], generator=g)
            x_train_all = x_train_all[perm]
            y_train_all = y_train_all[perm]

        ntrain = min(args.ntrain, x_train_all.shape[0])
        ntest = min(args.ntest, x_test_all.shape[0])
        x_train = x_train_all[:ntrain]
        y_train = y_train_all[:ntrain]
        x_test = x_test_all[:ntest]
        y_test = y_test_all[:ntest]

    x_train = x_train.unsqueeze(-1)
    x_test = x_test.unsqueeze(-1)
    return x_train, y_train, x_test, y_test


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="PDE-induced Random Feature Operator baseline for 1D Burgers-like datasets",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data args (compatible surface with fourier_1d style)
    p.add_argument("--data-mode", choices=["single_split", "separate_files"], default="single_split")
    p.add_argument("--data-file", type=str, default="data/burgers_data_R10.mat")
    p.add_argument("--train-file", type=str, default=None)
    p.add_argument("--test-file", type=str, default=None)
    p.add_argument("--train-split", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--shuffle", action="store_true")
    p.add_argument("--ntrain", type=int, default=1000)
    p.add_argument("--ntest", type=int, default=100)
    p.add_argument("--sub", type=int, default=2**3)

    # PDE-RF args
    p.add_argument("--M", type=int, default=2048)
    p.add_argument("--operator", choices=["heat", "advection", "convdiff", "wave"], default="heat")
    p.add_argument("--nu", type=float, default=1.0)
    p.add_argument("--tau-dist", choices=["loguniform", "uniform", "exponential"], default="loguniform")
    p.add_argument("--tau-min", type=float, default=1e-4)
    p.add_argument("--tau-max", type=float, default=1.0)
    p.add_argument("--tau-exp-rate", type=float, default=1.0)

    p.add_argument("--c-dist", choices=["uniform", "normal", "fixed"], default="uniform")
    p.add_argument("--c-max", type=float, default=1.0)
    p.add_argument("--c-std", type=float, default=1.0)
    p.add_argument("--c-fixed", type=float, default=1.0)

    p.add_argument("--wave-c-dist", choices=["uniform", "loguniform", "fixed"], default="uniform")
    p.add_argument("--wave-c-min", type=float, default=0.1)
    p.add_argument("--wave-c-max", type=float, default=2.0)
    p.add_argument("--wave-c-fixed", type=float, default=1.0)
    p.add_argument("--wave-gamma", type=float, default=0.0)

    p.add_argument("--g-smooth-tau", type=float, default=0.0)
    p.add_argument("--activation", choices=["tanh", "gelu", "relu", "sin"], default="tanh")
    p.add_argument("--feature-scale", choices=["none", "inv_sqrt_m"], default="inv_sqrt_m")
    p.add_argument("--ridge-lambda", type=float, default=1e-6)
    p.add_argument("--solve-device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    p.add_argument("--inner-product", choices=["mean", "sum"], default="mean")
    p.add_argument("--feature-batch-size", type=int, default=256)
    p.add_argument("--ridge-y-chunk", type=int, default=1024)
    p.add_argument("--viz-dir", type=str, default="visualizations/pde_rf_1d")
    p.add_argument("--num-viz", type=int, default=3)
    p.add_argument("--smoke-test", action="store_true")

    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.ridge_lambda <= 0:
        raise ValueError("--ridge-lambda must be > 0")
    if args.operator in {"advection", "convdiff"} and args.c_dist == "uniform" and args.c_max <= 0:
        raise ValueError("--c-max must be > 0")
    if args.wave_gamma < 0:
        raise ValueError("--wave-gamma must be >= 0")

    set_random_seed(args.seed)
    dtype = to_torch_dtype(args.dtype)
    device = _choose_device(args.solve_device)

    x_train, y_train, x_test, y_test = _load_data(args, dtype=dtype)
    ntrain, s = x_train.shape[0], x_train.shape[1]
    ntest = x_test.shape[0]

    x_train_in = x_train.squeeze(-1).to(device=device, dtype=dtype)
    x_test_in = x_test.squeeze(-1).to(device=device, dtype=dtype)
    y_train_t = y_train.to(device=device, dtype=dtype)
    y_test_t = y_test.to(device=device, dtype=dtype)

    feature_map = PDERandomFeatureMap1D(
        size=s,
        m=args.M,
        nu=args.nu,
        tau_dist=args.tau_dist,
        tau_min=args.tau_min,
        tau_max=args.tau_max,
        tau_exp_rate=args.tau_exp_rate,
        g_smooth_tau=args.g_smooth_tau,
        activation=args.activation,
        feature_scale=args.feature_scale,
        inner_product=args.inner_product,
        operator=args.operator,
        c_dist=args.c_dist,
        c_min=-args.c_max,
        c_max=args.c_max,
        c_std=args.c_std,
        c_fixed=args.c_fixed,
        wave_c_dist=args.wave_c_dist,
        wave_c_min=args.wave_c_min,
        wave_c_max=args.wave_c_max,
        wave_c_fixed=args.wave_c_fixed,
        wave_gamma=args.wave_gamma,
        dtype=dtype,
        device=device,
    )

    phi_train = _compute_feature_matrix(feature_map, x_train_in, args.feature_batch_size)
    phi_test = _compute_feature_matrix(feature_map, x_test_in, args.feature_batch_size)

    w_t = solve_ridge(
        phi_train,
        y_train_t,
        lam=args.ridge_lambda,
        method="cholesky",
        y_chunk_size=args.ridge_y_chunk,
    )

    pred_train = phi_train @ w_t
    pred_test = phi_test @ w_t

    lp = LpLoss(d=1, p=2, size_average=False)
    train_rel = lp(pred_train.reshape(ntrain, -1), y_train_t.reshape(ntrain, -1)).item() / ntrain
    test_rel = lp(pred_test.reshape(ntest, -1), y_test_t.reshape(ntest, -1)).item() / ntest

    metrics = RunMetrics(train_rel_l2=train_rel, test_rel_l2=test_rel, ntrain=ntrain, ntest=ntest, s=s)
    print("[metrics]", asdict(metrics))

    pred_test_cpu = pred_test.detach().cpu()
    y_test_cpu = y_test_t.detach().cpu()
    x_test_cpu = x_test.detach().cpu()

    os.makedirs(args.viz_dir, exist_ok=True)
    per_sample_err = [rel_l2(pred_test_cpu[i], y_test_cpu[i]) for i in range(ntest)]
    plot_error_histogram(per_sample_err, os.path.join(args.viz_dir, "test_relL2_hist"))

    x_grid = np.linspace(0.0, 1.0, s)
    for i in range(min(args.num_viz, ntest)):
        plot_1d_prediction(
            x=x_grid,
            gt=y_test_cpu[i],
            pred=pred_test_cpu[i],
            input_u0=x_test_cpu[i].squeeze(-1),
            out_path_no_ext=os.path.join(args.viz_dir, f"sample_{i:03d}"),
            title_prefix=f"sample {i}: ",
        )

    if args.smoke_test:
        smoke_rel = float(np.mean(per_sample_err)) if per_sample_err else float("nan")
        print(f"[smoke-test] mean test relL2={smoke_rel:.4f}")


if __name__ == "__main__":
    main()
