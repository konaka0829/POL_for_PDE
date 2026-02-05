"""rfm_1d.py

Random Feature Model (RFM) baseline for the 1D Burgers operator, implemented to
match the workflow of the POL_for_PDE FNO scripts.

Feature map: Fourier-space random features (Nelsen & Stuart, arXiv:2408.06526)
  - eq. (3.5)–(3.7) in the paper
Training: convex least-squares via normal equations (eq. 2.25), inner products in
L2 using composite trapezoid rule (eq. 4.4).

Usage (matches fourier_1d.py style)
------------------------------

Separate train/test files (default):

  python rfm_1d.py \
    --data-mode separate_files \
    --train-file data/burgers_data_R10_train.mat \
    --test-file  data/burgers_data_R10_test.mat \
    --ntrain 1000 --ntest 200 --sub 8 \
    --m 1024 --lambda 0.0

Single split file:

  python rfm_1d.py \
    --data-mode single_split \
    --data-file data/burgers_data_R10.mat \
    --ntrain 1000 --ntest 200 --sub 8 \
    --m 1024 --lambda 0.0
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch

from cli_utils import add_data_mode_args, add_split_args, validate_data_mode_args
from rfm_models import RFM1DConfig, RandomFeatureModel1D
from utilities3 import MatReader
from viz_utils import ensure_dir, plot_1d_prediction, plot_error_histogram


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RFM (random features) for 1D Burgers")

    # Data args (kept consistent with POL_for_PDE)
    add_data_mode_args(
        parser,
        default_data_mode="single_split",
        default_data_file="data/burgers_data_R10.mat",
        default_train_file=None,
        default_test_file=None,
    )
    parser.add_argument("--ntrain", type=int, default=1000)
    parser.add_argument("--ntest", type=int, default=200)
    parser.add_argument("--sub", type=int, default=8, help="Subsample factor (same as fourier_1d.py)")

    # RFM hyperparameters
    parser.add_argument("--m", type=int, default=1024, help="#random features")
    parser.add_argument(
        "--lambda",
        dest="lamreg",
        type=float,
        default=0.0,
        help="Ridge regularization strength (paper: Burgers uses 0)",
    )

    # Feature-map hyperparameters (paper defaults)
    parser.add_argument("--alpha-g", dest="al_g", type=float, default=2.0, help="GRF regularity α' (paper: 2)")
    parser.add_argument("--tau-g", dest="tau_g", type=float, default=5.0, help="GRF inverse length τ' (paper: 5)")
    parser.add_argument(
        "--nu-rf",
        dest="nu_rf",
        type=float,
        default=2.5e-3,
        help="Filter scale δ (paper: 0.0025)",
    )
    parser.add_argument(
        "--delta",
        dest="nu_rf",
        type=float,
        default=2.5e-3,
        help="Alias of --nu-rf (filter scale δ)",
    )
    parser.add_argument("--beta", dest="al_rf", type=float, default=4.0, help="Filter exponent β (paper: 4)")
    parser.add_argument("--sig-rf", dest="sig_rf", type=float, default=1.0, help="Feature scale")
    parser.add_argument(
        "--kmax",
        type=int,
        default=None,
        help="Truncate GRF Fourier modes (default: K/2)",
    )

    # Batching
    parser.add_argument("--batch-size", type=int, default=20, help="Training data batch size")
    parser.add_argument(
        "--feature-batch-size",
        type=int,
        default=64,
        help="Number of random features processed per chunk in training",
    )
    parser.add_argument(
        "--feature-batch-size-test",
        type=int,
        default=128,
        help="Number of random features processed per chunk in prediction",
    )

    # Misc
    add_split_args(parser, default_train_split=0.8, default_seed=0)
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument(
        "--viz-num-samples",
        type=int,
        default=3,
        help="How many test samples to visualize",
    )
    parser.add_argument(
        "--viz-dir",
        type=str,
        default="visualizations/rfm_1d",
        help="Output directory for figures",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    validate_data_mode_args(args, parser)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    dtype = torch.float32

    # ------------------------------------------------------------------
    # Load data (same logic as fourier_1d.py)
    # ------------------------------------------------------------------
    sub = int(args.sub)
    ntrain = int(args.ntrain)
    ntest = int(args.ntest)

    if args.data_mode == "single_split":
        reader = MatReader(args.data_file)
        x_data_full = reader.read_field("a")
        y_data_full = reader.read_field("u")
        K_fine = int(x_data_full.shape[-1])

        x_data = x_data_full[:, ::sub]
        y_data = y_data_full[:, ::sub]
        K = int(x_data.shape[-1])

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

        x_train_full = train_reader.read_field("a")
        y_train_full = train_reader.read_field("u")
        x_test_full = test_reader.read_field("a")
        y_test_full = test_reader.read_field("u")

        K_fine = int(x_train_full.shape[-1])

        x_train = x_train_full[:ntrain, ::sub]
        y_train = y_train_full[:ntrain, ::sub]
        x_test = x_test_full[-ntest:, ::sub]
        y_test = y_test_full[-ntest:, ::sub]

        K = int(x_train.shape[-1])

    print("==== Data ====")
    print(f"K_fine={K_fine}  K(subsampled)={K}  sub={args.sub}")
    print(f"ntrain={ntrain}  ntest={ntest}")
    print(f"x_train: {tuple(x_train.shape)}  y_train: {tuple(y_train.shape)}")

    # ------------------------------------------------------------------
    # Build + train RFM
    # ------------------------------------------------------------------
    cfg = RFM1DConfig(
        K=K,
        K_fine=K_fine,
        m=int(args.m),
        lamreg=float(args.lamreg),
        nu_rf=float(args.nu_rf),
        al_rf=float(args.al_rf),
        sig_rf=float(args.sig_rf),
        tau_g=float(args.tau_g),
        al_g=float(args.al_g),
        kmax=args.kmax,
        batch_size=int(args.batch_size),
        feature_batch_size=int(args.feature_batch_size),
        feature_batch_size_test=int(args.feature_batch_size_test),
        device=device,
        dtype=dtype,
    )
    model = RandomFeatureModel1D(cfg)

    print("==== RFM config ====")
    print(cfg)

    t0 = time.time()
    model.fit(x_train, y_train)
    t_train = time.time() - t0
    print(f"Training done in {t_train:.2f} sec")

    # ------------------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------------------
    with torch.no_grad():
        train_err = model.per_sample_relative_errors(x_train, y_train).detach().cpu().numpy()
        test_err = model.per_sample_relative_errors(x_test, y_test).detach().cpu().numpy()

    print("==== Errors (relative L2, trapezoid) ====")
    print(f"train: mean={train_err.mean():.6f}  median={np.median(train_err):.6f}")
    print(f"test : mean={test_err.mean():.6f}  median={np.median(test_err):.6f}")

    # ------------------------------------------------------------------
    # Visualizations (aligned with POL_for_PDE style)
    # ------------------------------------------------------------------
    ensure_dir(args.viz_dir)
    plot_error_histogram(test_err.tolist(), os.path.join(args.viz_dir, "error_hist"), title="RFM 1D test error")

    # Plot a few prediction curves
    ns = min(int(args.viz_num_samples), ntest)
    for i in range(ns):
        a_i = x_test[i]
        y_i = y_test[i]
        pred_i = model.predict(a_i).detach().cpu().numpy()
        plot_1d_prediction(
            x=None,
            gt=y_i.detach().cpu().numpy(),
            pred=pred_i,
            input_u0=a_i.detach().cpu().numpy(),
            out_path_no_ext=os.path.join(args.viz_dir, f"sample_{i:03d}"),
            title_prefix=f"RFM 1D | sample {i}",
        )

    # Save a tiny summary
    summary_path = os.path.join(args.viz_dir, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("RFM 1D Burgers\n")
        f.write(str(cfg) + "\n")
        f.write(f"train_mean_relL2 {train_err.mean():.8f}\n")
        f.write(f"test_mean_relL2  {test_err.mean():.8f}\n")
        f.write(f"train_seconds    {t_train:.3f}\n")
    print(f"Saved visualizations and summary to: {args.viz_dir}")


if __name__ == "__main__":
    main()
