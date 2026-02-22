import argparse
import os
from typing import Tuple

import numpy as np
import torch

from cli_utils import add_data_mode_args, add_split_args, validate_data_mode_args
from pde_features import PDERandomFeatureMap1D, apply_heat_semigroup_1d
from ridge import solve_ridge
from utilities3 import LpLoss, MatReader
from viz_utils import plot_1d_prediction, plot_error_histogram, rel_l2


def _dtype_from_name(name: str) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "float64":
        return torch.float64
    raise ValueError(f"Unsupported dtype: {name}")


def _select_solve_device(name: str) -> torch.device:
    name = name.lower()
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("--solve-device=cuda requested but CUDA is not available")
        return torch.device("cuda")
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raise ValueError(f"Unknown solve device: {name}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PDE-induced Random Features baseline for 1D Burgers")
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
    parser.add_argument("--sub", type=int, default=2**3, help="Subsampling rate for Burgers grid.")

    parser.add_argument("--M", type=int, default=2048, help="Number of random features.")
    parser.add_argument("--nu", type=float, default=1.0, help="Diffusion coefficient of heat semigroup.")
    parser.add_argument(
        "--tau-dist",
        choices=("loguniform", "uniform", "exponential"),
        default="loguniform",
        help="Distribution for random semigroup times.",
    )
    parser.add_argument("--tau-min", type=float, default=1e-4, help="Minimum tau.")
    parser.add_argument("--tau-max", type=float, default=1.0, help="Maximum tau.")
    parser.add_argument("--tau-exp-rate", type=float, default=1.0, help="Rate for exponential tau distribution.")
    parser.add_argument(
        "--g-smooth-tau",
        type=float,
        default=0.0,
        help="Optional smoothing time applied to random g before h=T(tau)g.",
    )
    parser.add_argument("--activation", choices=("tanh", "gelu", "relu", "sin"), default="tanh")
    parser.add_argument("--feature-scale", choices=("none", "inv_sqrt_m"), default="inv_sqrt_m")
    parser.add_argument("--ridge-lambda", type=float, default=1e-6, help="Ridge regularization lambda (>0).")
    parser.add_argument("--solve-device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--viz-dir", type=str, default="visualizations/pde_rf_1d")
    parser.add_argument("--num-viz", type=int, default=3, help="Number of test samples to visualize.")
    parser.add_argument("--smoke-test", action="store_true", help="Run synthetic smoke test without MAT files.")
    return parser


def _load_real_data_1d(args: argparse.Namespace) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    ntrain = args.ntrain
    ntest = args.ntest
    sub = args.sub
    s = 2**13 // sub

    if args.data_mode == "single_split":
        dataloader = MatReader(args.data_file)
        x_data = dataloader.read_field("a")[:, ::sub]
        y_data = dataloader.read_field("u")[:, ::sub]

        total = x_data.shape[0]
        indices = np.arange(total)
        if args.shuffle:
            np.random.shuffle(indices)
        split_idx = int(total * args.train_split)
        train_idx = indices[:split_idx]
        test_idx = indices[split_idx:]

        if ntrain > len(train_idx) or ntest > len(test_idx):
            raise ValueError(
                f"Not enough samples for ntrain={ntrain}, ntest={ntest} with train split {args.train_split} (total={total})."
            )

        train_idx = train_idx[:ntrain]
        test_idx = test_idx[:ntest]
        x_train = x_data[train_idx]
        y_train = y_data[train_idx]
        x_test = x_data[test_idx]
        y_test = y_data[test_idx]
    else:
        train_reader = MatReader(args.train_file)
        test_reader = MatReader(args.test_file)
        x_train = train_reader.read_field("a")[:ntrain, ::sub]
        y_train = train_reader.read_field("u")[:ntrain, ::sub]
        x_test = test_reader.read_field("a")[-ntest:, ::sub]
        y_test = test_reader.read_field("u")[-ntest:, ::sub]

    x_train = x_train.reshape(ntrain, s, 1)
    x_test = x_test.reshape(ntest, s, 1)
    return x_train, y_train, x_test, y_test, s


def _load_smoke_data_1d(dtype: torch.dtype, seed: int, ntrain: int = 64, ntest: int = 16, s: int = 128) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)

    x_train = torch.randn(ntrain, s, generator=g, dtype=dtype)
    x_test = torch.randn(ntest, s, generator=g, dtype=dtype)
    tau_true = torch.tensor(0.15, dtype=dtype)

    y_train = apply_heat_semigroup_1d(x_train, tau=tau_true, nu=1.0)
    y_test = apply_heat_semigroup_1d(x_test, tau=tau_true, nu=1.0)

    return x_train.unsqueeze(-1), y_train, x_test.unsqueeze(-1), y_test, s


def _build_features_1d(
    feature_map: PDERandomFeatureMap1D,
    x: torch.Tensor,
    batch_size: int = 512,
) -> torch.Tensor:
    x2 = x.squeeze(-1)
    out = []
    for start in range(0, x2.shape[0], batch_size):
        out.append(feature_map.features(x2[start : start + batch_size]))
    return torch.cat(out, dim=0)


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    validate_data_mode_args(args, parser)

    if args.ridge_lambda <= 0.0:
        parser.error("--ridge-lambda must be > 0")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    dtype = _dtype_from_name(args.dtype)
    solve_device = _select_solve_device(args.solve_device)

    if args.smoke_test:
        x_train, y_train, x_test, y_test, s = _load_smoke_data_1d(dtype=dtype, seed=args.seed)
        print("[smoke-test] Using synthetic data (1D heat semigroup).")
    else:
        x_train, y_train, x_test, y_test, s = _load_real_data_1d(args)

    ntrain = x_train.shape[0]
    ntest = x_test.shape[0]

    x_train = x_train.to(dtype=dtype)
    y_train = y_train.to(dtype=dtype)
    x_test = x_test.to(dtype=dtype)
    y_test = y_test.to(dtype=dtype)

    feature_map = PDERandomFeatureMap1D(
        S=s,
        M=args.M,
        nu=args.nu,
        tau_dist=args.tau_dist,
        tau_min=args.tau_min,
        tau_max=args.tau_max,
        tau_exp_rate=args.tau_exp_rate,
        g_smooth_tau=args.g_smooth_tau,
        activation=args.activation,
        feature_scale=args.feature_scale,
        inner_product="mean",
        device=solve_device,
        dtype=dtype,
    )

    Phi_train = _build_features_1d(feature_map, x_train.to(solve_device))
    Phi_test = _build_features_1d(feature_map, x_test.to(solve_device))
    Y_train = y_train.to(device=solve_device, dtype=dtype)

    W_T = solve_ridge(Phi_train, Y_train, lam=args.ridge_lambda, jitter=1e-10, method="cholesky", chunk_size=1024)

    yhat_test = (Phi_test @ W_T).to("cpu")

    myloss = LpLoss(d=1, size_average=False)
    test_rel = myloss(yhat_test.reshape(ntest, -1), y_test.reshape(ntest, -1)).item() / ntest
    per_sample_err = [rel_l2(yhat_test[i], y_test[i]) for i in range(ntest)]

    print(f"ntrain={ntrain} ntest={ntest} s={s} M={args.M} device={solve_device} dtype={args.dtype}")
    print(f"test relL2 (LpLoss avg): {test_rel:.6f}")
    print(f"test relL2 mean/median (viz_utils): {np.mean(per_sample_err):.6f} / {np.median(per_sample_err):.6f}")

    os.makedirs(args.viz_dir, exist_ok=True)
    plot_error_histogram(per_sample_err, os.path.join(args.viz_dir, "test_relL2_hist"))

    num_viz = max(0, min(args.num_viz, ntest))
    x_grid = np.linspace(0.0, 1.0, s)
    for i in range(num_viz):
        plot_1d_prediction(
            x=x_grid,
            gt=y_test[i],
            pred=yhat_test[i],
            input_u0=x_test[i].reshape(-1),
            out_path_no_ext=os.path.join(args.viz_dir, f"sample_{i:03d}"),
            title_prefix=f"sample {i}: ",
        )


if __name__ == "__main__":
    main()
