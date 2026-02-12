import argparse
import json
import os
from timeit import default_timer

import numpy as np
import torch
import torch.nn.functional as F

from cli_utils import add_data_mode_args, add_split_args, validate_data_mode_args
from rfm_burgers_1d import RandomFeatureModel
from utilities3 import LpLoss, MatReader
from viz_utils import plot_1d_prediction, plot_error_histogram, rel_l2


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Random Feature Model (RFM) for Burgers 1D")
    add_data_mode_args(
        parser,
        default_data_mode="single_split",
        default_data_file="data/burgers_data_R10.mat",
        default_train_file=None,
        default_test_file=None,
    )
    parser.add_argument("--ntrain", type=int, default=1000, help="Number of training samples.")
    parser.add_argument("--ntest", type=int, default=100, help="Number of test samples.")
    parser.add_argument("--sub", type=int, default=8, help="Uniform subsampling rate for both a and u.")
    parser.add_argument("--m", type=int, default=128, help="Number of random features.")
    parser.add_argument("--kmax", type=int, default=32, help="Max Fourier mode for random features (clamped to K//2).")
    parser.add_argument(
        "--lambda",
        dest="lambda_reg",
        type=float,
        default=1e-6,
        help="KRR regularization lambda. Internal solve uses lamreg = ntrain * lambda.",
    )
    parser.add_argument("--nu-rf", type=float, default=0.31946787, help="RF filter scale.")
    parser.add_argument("--al-rf", type=float, default=0.1, help="RF filter decay.")
    parser.add_argument("--sig-rf", type=float, default=2.597733, help="RF output scale.")
    parser.add_argument("--tau-g", type=float, default=15.045227, help="GRF inverse length-scale.")
    parser.add_argument("--al-g", type=float, default=2.9943917, help="GRF regularity exponent.")
    parser.add_argument(
        "--sig-g",
        type=float,
        default=None,
        help="GRF amplitude. If omitted, uses repo2-compatible auto setting: tau_g**(0.5*(2*al_g-1)).",
    )
    parser.add_argument("--bsize-train", type=int, default=25, help="Data batch size for forming normal equations.")
    parser.add_argument("--bsize-test", type=int, default=50, help="Test batch size (for compatibility/logging).")
    parser.add_argument("--bsize-grf-train", type=int, default=50, help="Feature batch size in training loops.")
    parser.add_argument("--bsize-grf-test", type=int, default=100, help="Feature batch size in prediction loops.")
    parser.add_argument("--bsize-grf-sample", type=int, default=128, help="Feature batch size when sampling GRFs.")
    parser.add_argument("--cpu", action="store_true", help="Force CPU execution.")
    add_split_args(parser, default_train_split=0.8, default_seed=0)
    return parser


def _validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    validate_data_mode_args(args, parser)
    if args.sub <= 0:
        parser.error("--sub must be positive.")
    if args.ntrain <= 0 or args.ntest <= 0:
        parser.error("--ntrain and --ntest must be positive.")


def _load_data(args: argparse.Namespace) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    if args.data_mode == "single_split":
        reader = MatReader(args.data_file)
        x_full = reader.read_field("a")
        y_full = reader.read_field("u")
        K_fine = int(x_full.shape[-1])

        x_data = x_full[:, :: args.sub]
        y_data = y_full[:, :: args.sub]
        total = x_data.shape[0]

        indices = np.arange(total)
        if args.shuffle:
            np.random.shuffle(indices)
        split_idx = int(total * args.train_split)
        train_idx = indices[:split_idx]
        test_idx = indices[split_idx:]
        if args.ntrain > len(train_idx) or args.ntest > len(test_idx):
            raise ValueError(
                f"Not enough samples for ntrain={args.ntrain}, ntest={args.ntest} "
                f"with train split {args.train_split} (total={total})."
            )
        train_idx = train_idx[: args.ntrain]
        test_idx = test_idx[: args.ntest]

        x_train = x_data[train_idx]
        y_train = y_data[train_idx]
        x_test = x_data[test_idx]
        y_test = y_data[test_idx]
    else:
        train_reader = MatReader(args.train_file)
        test_reader = MatReader(args.test_file)

        x_train_full = train_reader.read_field("a")
        y_train_full = train_reader.read_field("u")
        x_test_full = test_reader.read_field("a")
        y_test_full = test_reader.read_field("u")
        K_fine = int(x_train_full.shape[-1])

        if args.ntrain > x_train_full.shape[0] or args.ntest > x_test_full.shape[0]:
            raise ValueError(
                f"Not enough samples in separate files for ntrain={args.ntrain}, ntest={args.ntest} "
                f"(train_total={x_train_full.shape[0]}, test_total={x_test_full.shape[0]})."
            )

        x_train = x_train_full[: args.ntrain, :: args.sub]
        y_train = y_train_full[: args.ntrain, :: args.sub]
        x_test = x_test_full[-args.ntest :, :: args.sub]
        y_test = y_test_full[-args.ntest :, :: args.sub]

    return x_train, y_train, x_test, y_test, K_fine


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    _validate_args(args, parser)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu"))
    x_train, y_train, x_test, y_test, K_fine = _load_data(args)

    K = int(x_train.shape[-1])
    if K % 2 != 0:
        raise ValueError(f"Subsampled spatial size K must be even for FFT, but got K={K}.")
    if x_test.shape[-1] != K:
        raise ValueError("Train/test spatial resolutions do not match after subsampling.")

    kmax = min(args.kmax, K // 2)
    if kmax != args.kmax:
        print(f"[info] clamped kmax from {args.kmax} to {kmax} (K//2).")
    lamreg = args.ntrain * args.lambda_reg

    model = RandomFeatureModel(
        K=K,
        n=args.ntrain,
        m=args.m,
        ntest=args.ntest,
        lamreg=lamreg,
        nu_rf=args.nu_rf,
        al_rf=args.al_rf,
        bsize_train=args.bsize_train,
        bsize_test=args.bsize_test,
        bsize_grf_train=args.bsize_grf_train,
        bsize_grf_test=args.bsize_grf_test,
        bsize_grf_sample=args.bsize_grf_sample,
        device=device,
        al_g=args.al_g,
        tau_g=args.tau_g,
        sig_g=args.sig_g,
        kmax=kmax,
        sig_rf=args.sig_rf,
        K_fine=K_fine,
    )

    model.load_train(x_train, y_train)
    model.load_test(x_test, y_test)
    model.output_train_noisy = y_train

    t0 = default_timer()
    model.fit()
    fit_sec = default_timer() - t0

    with torch.no_grad():
        pred_train = model.predict(x_train).cpu()
        pred_test = model.predict(x_test).cpu()

    myloss = LpLoss(size_average=False)
    train_rel_l2 = myloss(pred_train, y_train).item() / args.ntrain
    test_rel_l2 = myloss(pred_test, y_test).item() / args.ntest
    train_mse = F.mse_loss(pred_train, y_train, reduction="mean").item()
    test_mse = F.mse_loss(pred_test, y_test, reduction="mean").item()

    print("RFM 1D (Burgers)")
    print(f"device={device} K={K} K_fine={K_fine} ntrain={args.ntrain} ntest={args.ntest}")
    print(f"m={args.m} kmax={kmax} lambda={args.lambda_reg} lamreg={lamreg}")
    print(f"fit_time_sec={fit_sec:.3f}")
    print(f"train_relL2={train_rel_l2:.6e} test_relL2={test_rel_l2:.6e}")
    print(f"train_mse={train_mse:.6e} test_mse={test_mse:.6e}")

    viz_dir = os.path.join("visualizations", "rfm_1d")
    os.makedirs(viz_dir, exist_ok=True)

    per_sample_err = [rel_l2(pred_test[i], y_test[i]) for i in range(pred_test.shape[0])]
    plot_error_histogram(per_sample_err, os.path.join(viz_dir, "test_relL2_hist"))

    sample_ids = []
    for sid in (0, 1, 2):
        sid = min(sid, args.ntest - 1)
        if sid not in sample_ids:
            sample_ids.append(sid)
    x_grid = np.linspace(0.0, 1.0, K)
    for sid in sample_ids:
        plot_1d_prediction(
            x=x_grid,
            gt=y_test[sid],
            pred=pred_test[sid],
            out_path_no_ext=os.path.join(viz_dir, f"sample_{sid:03d}"),
            input_u0=x_test[sid],
            title_prefix=f"sample {sid}: ",
        )

    np.save(os.path.join(viz_dir, "al_model.npy"), model.al_model.detach().cpu().numpy())
    np.save(os.path.join(viz_dir, "grf_g.npy"), model.grf_g.detach().cpu().numpy())

    metrics = {
        "train_rel_l2": train_rel_l2,
        "test_rel_l2": test_rel_l2,
        "train_mse": train_mse,
        "test_mse": test_mse,
        "fit_time_sec": fit_sec,
        "device": str(device),
        "data_mode": args.data_mode,
        "data_file": args.data_file,
        "train_file": args.train_file,
        "test_file": args.test_file,
        "train_split": args.train_split,
        "shuffle": args.shuffle,
        "seed": args.seed,
        "sub": args.sub,
        "ntrain": args.ntrain,
        "ntest": args.ntest,
        "K": K,
        "K_fine": K_fine,
        "m": args.m,
        "kmax": kmax,
        "lambda": args.lambda_reg,
        "lamreg": lamreg,
        "nu_rf": args.nu_rf,
        "al_rf": args.al_rf,
        "sig_rf": args.sig_rf,
        "tau_g": args.tau_g,
        "al_g": args.al_g,
        "sig_g": model.sig_g,
        "bsize_train": args.bsize_train,
        "bsize_test": args.bsize_test,
        "bsize_grf_train": args.bsize_grf_train,
        "bsize_grf_test": args.bsize_grf_test,
        "bsize_grf_sample": args.bsize_grf_sample,
    }
    with open(os.path.join(viz_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
