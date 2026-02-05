"""RFM baseline for 1D Burgers (random feature model)."""

from __future__ import annotations

import argparse
from timeit import default_timer
from typing import List

import numpy as np
import torch

from cli_utils import add_data_mode_args, add_split_args, validate_data_mode_args
from rfm_core import fit_rfm, predict_from_features, save_rfm_model
from rfm_features import BurgersRFFeatures, grf_sample_1d
from utilities3 import LpLoss, MatReader
from viz_utils import (
    LearningCurve,
    ensure_dir,
    plot_1d_prediction,
    plot_error_histogram,
    plot_learning_curve,
    rel_l2,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RFM 1D (Burgers)")
    add_data_mode_args(
        parser,
        default_data_mode="single_split",
        default_data_file="data/burgers_data_R10.mat",
        default_train_file=None,
        default_test_file=None,
    )
    parser.add_argument("--ntrain", type=int, default=1000, help="Number of training samples.")
    parser.add_argument("--ntest", type=int, default=100, help="Number of test samples.")
    parser.add_argument("--sub", type=int, default=2**3, help="Subsampling rate.")
    parser.add_argument("--batch-size", type=int, default=20, help="Batch size.")
    parser.add_argument("--m", type=int, default=1024, help="Number of random features.")
    parser.add_argument("--lam", type=float, default=0.0, help="Ridge regularization lambda.")
    parser.add_argument("--delta", type=float, default=0.0025, help="Burgers filter delta.")
    parser.add_argument("--beta", type=float, default=4.0, help="Burgers filter beta.")
    parser.add_argument("--tau-theta", type=float, default=5.0, help="GRF tau.")
    parser.add_argument("--alpha-theta", type=float, default=2.0, help="GRF alpha.")
    parser.add_argument("--rf-seed", type=int, default=0, help="Random feature seed.")
    parser.add_argument("--device", type=str, default=None, help="Device (e.g. cuda or cpu).")
    parser.add_argument("--save-model", action="store_true", help="Save fitted model.")
    parser.add_argument("--model-out", type=str, default="model/rfm_1d.pt", help="Model output path.")
    add_split_args(parser, default_train_split=0.8, default_seed=0)
    return parser


def _validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    validate_data_mode_args(args, parser)


def _select_device(device_arg: str | None) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_data(args: argparse.Namespace) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    sub = args.sub
    h = 2**13 // sub
    s = h
    ntrain = args.ntrain
    ntest = args.ntest

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
                f"Not enough samples for ntrain={ntrain}, ntest={ntest} with train split "
                f"{args.train_split} (total={total})."
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
    theta = grf_sample_1d(
        args.m,
        s,
        tau=args.tau_theta,
        alpha=args.alpha_theta,
        device=device,
        generator=generator,
    )
    features = BurgersRFFeatures(theta=theta, delta=args.delta, beta=args.beta)

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
                pred = predict_from_features(alpha, phi, args.m)
                total_l2 += myloss(pred, y).item()
                per_sample_errors.extend(per_sample_loss(pred, y).cpu().numpy().tolist())
        return total_l2, per_sample_errors

    train_l2_sum, _ = _eval(train_loader)
    test_l2_sum, test_errors = _eval(test_loader)
    train_rel = train_l2_sum / ntrain
    test_rel = test_l2_sum / ntest
    print(f"Train rel L2: {train_rel:.4f}")
    print(f"Test rel L2: {test_rel:.4f}")

    viz_dir = "visualizations/rfm_1d"
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
        title="rfm_1d: relative L2",
    )

    plot_error_histogram(
        test_errors,
        out_path_no_ext=f"{viz_dir}/error_hist",
        title="rfm_1d: test relative L2 histogram",
    )

    sample_ids = [0, min(1, ntest - 1), min(2, ntest - 1)]
    x_grid = torch.linspace(0, 1, s)
    with torch.no_grad():
        for idx in sample_ids:
            a = x_test[idx : idx + 1].to(device)
            y = y_test[idx : idx + 1].to(device)
            phi = features(a)
            pred = predict_from_features(alpha, phi, args.m)
            plot_1d_prediction(
                x_grid,
                gt=y.squeeze().cpu(),
                pred=pred.squeeze().cpu(),
                out_path_no_ext=f"{viz_dir}/sample_{idx}",
                input_u0=a.squeeze().cpu(),
                title_prefix=f"rfm_1d sample {idx} (relL2={rel_l2(pred, y):.3g})",
            )

    if args.save_model:
        ensure_dir("model")
        save_rfm_model(
            args.model_out,
            alpha=alpha,
            theta={"theta": theta},
            hyperparams={
                "m": float(args.m),
                "lam": float(args.lam),
                "delta": float(args.delta),
                "beta": float(args.beta),
                "tau_theta": float(args.tau_theta),
                "alpha_theta": float(args.alpha_theta),
            },
        )
        print(f"Saved model to {args.model_out}")


if __name__ == "__main__":
    main()
