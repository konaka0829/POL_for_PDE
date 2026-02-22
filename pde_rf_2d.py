#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from dataclasses import asdict, dataclass

import numpy as np
import torch

from basis import GridBasis, PODBasis
from pde_features import PDERandomFeatureMap2D, apply_heat_semigroup_2d, set_random_seed, to_torch_dtype
from ridge import solve_ridge
from utilities3 import LpLoss, MatReader, UnitGaussianNormalizer
from viz_utils import plot_2d_comparison, plot_error_histogram, rel_l2


@dataclass
class RunMetrics:
    train_rel_l2: float
    test_rel_l2: float
    ntrain: int
    ntest: int
    s: int
    basis: str


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
    feature_map: PDERandomFeatureMap2D,
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
        s = 32
        ntrain = min(args.ntrain, 256)
        ntest = min(args.ntest, 64)
        total = ntrain + ntest

        a = torch.randn(total, s, s, dtype=dtype)
        tau_true = 0.08
        u = apply_heat_semigroup_2d(a, tau=tau_true, nu=args.nu)

        x_train = a[:ntrain]
        y_train = u[:ntrain]
        x_test = a[ntrain : ntrain + ntest]
        y_test = u[ntrain : ntrain + ntest]
        return x_train, y_train, x_test, y_test

    s = int(((args.grid_size - 1) / args.r) + 1)

    if args.data_mode == "single_split":
        if not args.data_file:
            raise ValueError("--data-file is required for --data-mode single_split")
        reader = MatReader(args.data_file)
        x_data = reader.read_field("coeff")[:, :: args.r, :: args.r][:, :s, :s].to(dtype=dtype)
        y_data = reader.read_field("sol")[:, :: args.r, :: args.r][:, :s, :s].to(dtype=dtype)

        n_total = x_data.shape[0]
        ntrain = min(args.ntrain, n_total)
        ntest = min(args.ntest, n_total - ntrain)
        if ntest <= 0:
            raise ValueError("No test data available; adjust ntrain/ntest")

        x_train = x_data[:ntrain]
        y_train = y_data[:ntrain]
        x_test = x_data[ntrain : ntrain + ntest]
        y_test = y_data[ntrain : ntrain + ntest]

    else:
        if not args.train_file or not args.test_file:
            raise ValueError("--train-file and --test-file are required for --data-mode separate_files")

        train_reader = MatReader(args.train_file)
        test_reader = MatReader(args.test_file)

        x_train_all = train_reader.read_field("coeff")[:, :: args.r, :: args.r][:, :s, :s].to(dtype=dtype)
        y_train_all = train_reader.read_field("sol")[:, :: args.r, :: args.r][:, :s, :s].to(dtype=dtype)
        x_test_all = test_reader.read_field("coeff")[:, :: args.r, :: args.r][:, :s, :s].to(dtype=dtype)
        y_test_all = test_reader.read_field("sol")[:, :: args.r, :: args.r][:, :s, :s].to(dtype=dtype)

        ntrain = min(args.ntrain, x_train_all.shape[0])
        ntest = min(args.ntest, x_test_all.shape[0])

        x_train = x_train_all[:ntrain]
        y_train = y_train_all[:ntrain]
        x_test = x_test_all[:ntest]
        y_test = y_test_all[:ntest]

    return x_train, y_train, x_test, y_test


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="PDE-induced Random Feature Operator baseline for 2D Darcy-like datasets",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data args
    p.add_argument("--data-mode", choices=["single_split", "separate_files"], default="separate_files")
    p.add_argument("--data-file", type=str, default=None)
    p.add_argument("--train-file", type=str, default="data/piececonst_r421_N1024_smooth1.mat")
    p.add_argument("--test-file", type=str, default="data/piececonst_r421_N1024_smooth2.mat")
    p.add_argument("--ntrain", type=int, default=1000)
    p.add_argument("--ntest", type=int, default=100)
    p.add_argument("--r", type=int, default=5)
    p.add_argument("--grid-size", type=int, default=421)
    p.add_argument("--seed", type=int, default=0)

    # PDE-RF args
    p.add_argument("--M", type=int, default=2048)
    p.add_argument("--nu", type=float, default=1.0)
    p.add_argument("--tau-dist", choices=["loguniform", "uniform", "exponential"], default="loguniform")
    p.add_argument("--tau-min", type=float, default=1e-4)
    p.add_argument("--tau-max", type=float, default=1.0)
    p.add_argument("--tau-exp-rate", type=float, default=1.0)
    p.add_argument("--g-smooth-tau", type=float, default=0.0)
    p.add_argument("--activation", choices=["tanh", "gelu", "relu", "sin"], default="tanh")
    p.add_argument("--feature-scale", choices=["none", "inv_sqrt_m"], default="inv_sqrt_m")
    p.add_argument("--ridge-lambda", type=float, default=1e-6)
    p.add_argument("--solve-device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    p.add_argument("--inner-product", choices=["mean", "sum"], default="mean")
    p.add_argument("--feature-batch-size", type=int, default=128)
    p.add_argument("--ridge-y-chunk", type=int, default=1024)

    p.add_argument("--basis", choices=["grid", "pod"], default="grid")
    p.add_argument("--basis-dim", type=int, default=256)
    p.add_argument("--pod-center", dest="pod_center", action="store_true")
    p.add_argument("--no-pod-center", dest="pod_center", action="store_false")
    p.set_defaults(pod_center=True)

    p.add_argument("--viz-dir", type=str, default="visualizations/pde_rf_2d")
    p.add_argument("--num-viz", type=int, default=3)
    p.add_argument("--smoke-test", action="store_true")

    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.ridge_lambda <= 0:
        raise ValueError("--ridge-lambda must be > 0")

    set_random_seed(args.seed)
    dtype = to_torch_dtype(args.dtype)
    device = _choose_device(args.solve_device)

    x_train_raw, y_train_raw, x_test_raw, y_test_raw = _load_data(args, dtype=dtype)
    ntrain, s = x_train_raw.shape[0], x_train_raw.shape[1]
    ntest = x_test_raw.shape[0]

    x_train_raw = x_train_raw.to(device=device, dtype=dtype)
    y_train_raw = y_train_raw.to(device=device, dtype=dtype)
    x_test_raw = x_test_raw.to(device=device, dtype=dtype)
    y_test_raw = y_test_raw.to(device=device, dtype=dtype)

    x_normalizer = UnitGaussianNormalizer(x_train_raw).to(device)
    y_normalizer = UnitGaussianNormalizer(y_train_raw).to(device)

    x_train_norm = x_normalizer.encode(x_train_raw)
    x_test_norm = x_normalizer.encode(x_test_raw)
    y_train_norm = y_normalizer.encode(y_train_raw)

    feature_map = PDERandomFeatureMap2D(
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
        dtype=dtype,
        device=device,
    )

    phi_train = _compute_feature_matrix(feature_map, x_train_norm, args.feature_batch_size)
    phi_test = _compute_feature_matrix(feature_map, x_test_norm, args.feature_batch_size)

    y_train_flat = y_train_norm.reshape(ntrain, -1)
    if args.basis == "grid":
        basis = GridBasis(spatial_shape=(s, s))
        coeff_train = basis.encode(y_train_norm)
    else:
        basis = PODBasis.fit(y_train_flat, basis_dim=args.basis_dim, center=args.pod_center)
        coeff_train = basis.encode(y_train_flat)

    w_t = solve_ridge(
        phi_train,
        coeff_train,
        lam=args.ridge_lambda,
        method="cholesky",
        y_chunk_size=args.ridge_y_chunk,
    )

    coeff_train_hat = phi_train @ w_t
    coeff_test_hat = phi_test @ w_t

    if args.basis == "grid":
        yhat_train_norm = basis.decode(coeff_train_hat)
        yhat_test_norm = basis.decode(coeff_test_hat)
    else:
        yhat_train_norm = basis.decode(coeff_train_hat).reshape(ntrain, s, s)
        yhat_test_norm = basis.decode(coeff_test_hat).reshape(ntest, s, s)

    yhat_train = y_normalizer.decode(yhat_train_norm)
    yhat_test = y_normalizer.decode(yhat_test_norm)

    lp = LpLoss(d=2, p=2, size_average=False)
    train_rel = lp(yhat_train.reshape(ntrain, -1), y_train_raw.reshape(ntrain, -1)).item() / ntrain
    test_rel = lp(yhat_test.reshape(ntest, -1), y_test_raw.reshape(ntest, -1)).item() / ntest

    metrics = RunMetrics(
        train_rel_l2=train_rel,
        test_rel_l2=test_rel,
        ntrain=ntrain,
        ntest=ntest,
        s=s,
        basis=args.basis,
    )
    print("[metrics]", asdict(metrics))

    pred_cpu = yhat_test.detach().cpu()
    gt_cpu = y_test_raw.detach().cpu()
    x_cpu = x_test_raw.detach().cpu()

    os.makedirs(args.viz_dir, exist_ok=True)
    per_sample_err = [rel_l2(pred_cpu[i], gt_cpu[i]) for i in range(ntest)]
    plot_error_histogram(per_sample_err, os.path.join(args.viz_dir, "test_relL2_hist"))

    for i in range(min(args.num_viz, ntest)):
        plot_2d_comparison(
            gt=gt_cpu[i],
            pred=pred_cpu[i],
            input_field=x_cpu[i],
            out_path_no_ext=os.path.join(args.viz_dir, f"sample_{i:03d}"),
            suptitle=f"sample {i}  relL2={per_sample_err[i]:.3g}",
        )

    if args.smoke_test:
        smoke_rel = float(np.mean(per_sample_err)) if per_sample_err else float("nan")
        print(f"[smoke-test] mean test relL2={smoke_rel:.4f}")


if __name__ == "__main__":
    main()
