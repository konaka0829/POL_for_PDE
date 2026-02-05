"""RFM baseline for 2D Darcy flow (random feature model)."""

from __future__ import annotations

import argparse
from timeit import default_timer
from typing import List

import numpy as np
import torch

from cli_utils import add_data_mode_args, validate_data_mode_args
from rfm_core import fit_rfm, predict_from_features, save_rfm_model
from rfm_features import DarcyRFFeatures, grf_sample_2d
from utilities3 import LpLoss, MatReader
from viz_utils import (
    LearningCurve,
    ensure_dir,
    plot_2d_comparison,
    plot_error_histogram,
    plot_learning_curve,
    rel_l2,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RFM 2D (Darcy)")
    add_data_mode_args(
        parser,
        default_data_mode="single_split",
        default_data_file="data/darcy_data.mat",
        default_train_file=None,
        default_test_file=None,
    )
    parser.add_argument("--ntrain", type=int, default=1000, help="Number of training samples.")
    parser.add_argument("--ntest", type=int, default=100, help="Number of test samples.")
    parser.add_argument("--batch-size", type=int, default=10, help="Batch size.")
    parser.add_argument("--r", type=int, default=1, help="Downsampling rate.")
    parser.add_argument("--grid-size", type=int, default=421, help="Original grid size.")
    parser.add_argument("--m", type=int, default=350, help="Number of random features.")
    parser.add_argument("--lam", type=float, default=1e-8, help="Ridge regularization lambda.")
    parser.add_argument("--tau-theta", type=float, default=7.5, help="GRF tau.")
    parser.add_argument("--alpha-theta", type=float, default=2.0, help="GRF alpha.")
    parser.add_argument("--s-plus", type=float, default=1 / 12, help="Sigma upper bound.")
    parser.add_argument("--s-minus", type=float, default=-1 / 3, help="Sigma lower bound.")
    parser.add_argument("--delta-sig", type=float, default=0.15, help="Sigma delta.")
    parser.add_argument("--eta", type=float, default=1e-4, help="Heat smoothing eta.")
    parser.add_argument("--dt", type=float, default=0.03, help="Heat smoothing dt.")
    parser.add_argument("--heat-steps", type=int, default=34, help="Heat smoothing steps.")
    parser.add_argument("--f-const", type=float, default=1.0, help="Darcy forcing constant.")
    parser.add_argument("--feature-chunk", type=int, default=32, help="Feature chunk size for Poisson solves.")
    parser.add_argument("--rf-seed", type=int, default=0, help="Random feature seed.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for shuffling.")
    parser.add_argument("--device", type=str, default=None, help="Device (e.g. cuda or cpu).")
    parser.add_argument("--save-model", action="store_true", help="Save fitted model.")
    parser.add_argument("--model-out", type=str, default="model/rfm_2d.pt", help="Model output path.")
    return parser


def _validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    validate_data_mode_args(args, parser)


def _select_device(device_arg: str | None) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_data(args: argparse.Namespace) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    ntrain = args.ntrain
    ntest = args.ntest
    r = args.r
    h = int(((args.grid_size - 1) / r) + 1)
    s = h

    if args.data_mode == "single_split":
        reader = MatReader(args.data_file)
        x_data = reader.read_field("coeff")[:, ::r, ::r][:, :s, :s]
        y_data = reader.read_field("sol")[:, ::r, ::r][:, :s, :s]
        x_train = x_data[:ntrain]
        y_train = y_data[:ntrain]
        x_test = x_data[-ntest:]
        y_test = y_data[-ntest:]
    else:
        reader = MatReader(args.train_file)
        x_train = reader.read_field("coeff")[:ntrain, ::r, ::r][:, :s, :s]
        y_train = reader.read_field("sol")[:ntrain, ::r, ::r][:, :s, :s]
        reader.load_file(args.test_file)
        x_test = reader.read_field("coeff")[:ntest, ::r, ::r][:, :s, :s]
        y_test = reader.read_field("sol")[:ntest, ::r, ::r][:, :s, :s]

    return x_train, y_train, x_test, y_test


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    _validate_args(args, parser)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = _select_device(args.device)

    x_train, y_train, x_test, y_test = _load_data(args)
    ntrain = args.ntrain
    ntest = args.ntest
    s = x_train.shape[-1]

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_train, y_train),
        batch_size=args.batch_size,
        shuffle=True,
    )
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_test, y_test),
        batch_size=args.batch_size,
        shuffle=False,
    )

    generator = torch.Generator(device=device).manual_seed(args.rf_seed)
    theta1 = grf_sample_2d(args.m, s, tau=args.tau_theta, alpha=args.alpha_theta, device=device, generator=generator)
    theta2 = grf_sample_2d(args.m, s, tau=args.tau_theta, alpha=args.alpha_theta, device=device, generator=generator)

    features = DarcyRFFeatures(
        theta1=theta1,
        theta2=theta2,
        s_plus=args.s_plus,
        s_minus=args.s_minus,
        delta_sig=args.delta_sig,
        eta=args.eta,
        dt=args.dt,
        heat_steps=args.heat_steps,
        f_const=args.f_const,
        feature_chunk_size=args.feature_chunk,
    )

    t0 = default_timer()
    alpha = fit_rfm(train_loader, features, args.m, args.lam, device)
    t1 = default_timer()
    print(f"Fit complete in {t1 - t0:.2f}s")

    myloss = LpLoss(size_average=False)
    per_sample_loss = LpLoss(reduction=False)

    def _eval(loader: torch.utils.data.DataLoader) -> tuple[float, List[float]]:
        total_l2 = 0.0
        per_sample_errors: List[float] = []
        with torch.no_grad():
            for a, y in loader:
                a = a.to(device)
                y = y.to(device)
                phi = features(a)
                pred = predict_from_features(alpha, phi.reshape(phi.shape[0], phi.shape[1], -1), args.m)
                pred = pred.reshape(y.shape)
                total_l2 += myloss(pred.reshape(pred.shape[0], -1), y.reshape(y.shape[0], -1)).item()
                per_sample_errors.extend(
                    per_sample_loss(pred.reshape(pred.shape[0], -1), y.reshape(y.shape[0], -1)).cpu().numpy().tolist()
                )
        return total_l2, per_sample_errors

    train_l2_sum, _ = _eval(train_loader)
    test_l2_sum, test_errors = _eval(test_loader)
    train_rel = train_l2_sum / ntrain
    test_rel = test_l2_sum / ntest
    print(f"Train rel L2: {train_rel:.4f}")
    print(f"Test rel L2: {test_rel:.4f}")

    viz_dir = "visualizations/rfm_2d"
    ensure_dir(viz_dir)

    plot_learning_curve(
        LearningCurve(
            epochs=[0],
            train=[train_rel],
            test=[test_rel],
            train_label="train (relL2)",
            test_label="test (relL2)",
            metric_name="relative L2",
        ),
        out_path_no_ext=f"{viz_dir}/learning_curve_relL2",
        logy=True,
        title="rfm_2d: relative L2",
    )

    plot_error_histogram(
        test_errors,
        out_path_no_ext=f"{viz_dir}/error_hist",
        title="rfm_2d: test relative L2 histogram",
    )

    sample_ids = [0, min(1, ntest - 1), min(2, ntest - 1)]
    with torch.no_grad():
        for idx in sample_ids:
            a = x_test[idx : idx + 1].to(device)
            y = y_test[idx : idx + 1].to(device)
            phi = features(a)
            pred = predict_from_features(alpha, phi.reshape(phi.shape[0], phi.shape[1], -1), args.m)
            pred = pred.reshape(y.shape)
            plot_2d_comparison(
                gt=y.squeeze().cpu(),
                pred=pred.squeeze().cpu(),
                input_field=a.squeeze().cpu(),
                out_path_no_ext=f"{viz_dir}/sample_{idx}",
                suptitle=f"rfm_2d sample {idx} (relL2={rel_l2(pred, y):.3g})",
            )

    if args.save_model:
        ensure_dir("model")
        save_rfm_model(
            args.model_out,
            alpha=alpha,
            theta={"theta1": theta1, "theta2": theta2},
            hyperparams={
                "m": float(args.m),
                "lam": float(args.lam),
                "tau_theta": float(args.tau_theta),
                "alpha_theta": float(args.alpha_theta),
                "s_plus": float(args.s_plus),
                "s_minus": float(args.s_minus),
                "delta_sig": float(args.delta_sig),
                "eta": float(args.eta),
                "dt": float(args.dt),
                "heat_steps": float(args.heat_steps),
                "f_const": float(args.f_const),
                "feature_chunk_size": float(args.feature_chunk),
            },
        )
        print(f"Saved model to {args.model_out}")


if __name__ == "__main__":
    main()
