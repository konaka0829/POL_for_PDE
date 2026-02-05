"""rfm_2d.py

Random Feature Model (RFM) baseline for the 2D Darcy flow operator, implemented
to match the workflow of the POL_for_PDE FNO scripts.

Feature map: predictor-corrector random features (Nelsen & Stuart, arXiv:2408.06526)
  - eq. (3.12)–(3.14) in the paper
Training: convex least-squares via normal equations (eq. 2.25), inner products in
L2 using composite trapezoid rule (eq. 4.4).

Usage (matches fourier_2d.py style)
------------------------------

Separate train/test files (default):

  python rfm_2d.py \
    --data-mode separate_files \
    --train-file data/piececonst_r421_N1024_smooth1.mat \
    --test-file  data/piececonst_r421_N1024_smooth2.mat \
    --ntrain 1000 --ntest 100 --r 5 --grid-size 421 \
    --m 256 --lambda 1e-8

Single split file:

  python rfm_2d.py \
    --data-mode single_split \
    --data-file data/piececonst_r421_N1024_smooth1.mat \
    --ntrain 1000 --ntest 100 --r 5 --grid-size 421 \
    --m 256 --lambda 1e-8
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch

from cli_utils import add_data_mode_args, add_split_args, validate_data_mode_args
from rfm_models import PredictorCorrectorRFM2D, RFM2DConfig
from utilities3 import MatReader
from viz_utils import ensure_dir, plot_2d_comparison, plot_error_histogram


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RFM (random features) for 2D Darcy flow")

    # Data args (kept consistent with fourier_2d.py)
    add_data_mode_args(
        parser,
        default_data_mode="separate_files",
        default_data_file="data/piececonst_r421_N1024_smooth1.mat",
        default_train_file="data/piececonst_r421_N1024_smooth1.mat",
        default_test_file="data/piececonst_r421_N1024_smooth2.mat",
    )

    parser.add_argument("--ntrain", type=int, default=1000, help="Number of training samples")
    parser.add_argument("--ntest", type=int, default=100, help="Number of test samples")
    parser.add_argument("--r", type=int, default=5, help="Downsampling rate (same as fourier_2d.py)")
    parser.add_argument("--grid-size", type=int, default=421, help="Original grid size (same as fourier_2d.py)")

    # RFM hyperparameters
    parser.add_argument("--m", type=int, default=256, help="#random features")
    parser.add_argument(
        "--lambda",
        dest="lamreg",
        type=float,
        default=1e-8,
        help="Ridge regularization strength (paper: Darcy uses 1e-8)",
    )

    # Feature-map hyperparameters (paper defaults)
    parser.add_argument("--alpha-g", dest="al_g", type=float, default=2.0, help="GRF regularity α' (paper: 2)")
    parser.add_argument("--tau-g", dest="tau_g", type=float, default=7.5, help="GRF inverse length τ' (paper: 7.5)")
    parser.add_argument("--s-plus", dest="s_plus", type=float, default=1.0 / 12.0)
    parser.add_argument("--s-minus", dest="s_minus", type=float, default=-1.0 / 3.0)
    parser.add_argument("--delta-sig", dest="delta_sig", type=float, default=0.15)

    # Smoothing (paper defaults)
    parser.add_argument("--smooth-dt", type=float, default=0.03)
    parser.add_argument("--smooth-eta", type=float, default=1e-4)
    parser.add_argument("--smooth-steps", type=int, default=34)

    # Batching
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--feature-batch-size", type=int, default=8)
    parser.add_argument("--feature-batch-size-test", type=int, default=16)

    # Misc
    add_split_args(parser, default_train_split=0.8, default_seed=0)
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument("--viz-num-samples", type=int, default=3)
    parser.add_argument("--viz-dir", type=str, default="visualizations/rfm_2d")
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
    # Load data (same logic as fourier_2d.py)
    # ------------------------------------------------------------------
    ntrain = int(args.ntrain)
    ntest = int(args.ntest)
    r = int(args.r)
    grid_size = int(args.grid_size)
    s = int(((grid_size - 1) / r) + 1)

    if args.data_mode == "single_split":
        reader = MatReader(args.data_file)
        x_data = reader.read_field("coeff")[:, ::r, ::r][:, :s, :s]
        y_data = reader.read_field("sol")[:, ::r, ::r][:, :s, :s]

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

        x_train = train_reader.read_field("coeff")[:ntrain, ::r, ::r][:, :s, :s]
        y_train = train_reader.read_field("sol")[:ntrain, ::r, ::r][:, :s, :s]
        x_test = test_reader.read_field("coeff")[-ntest:, ::r, ::r][:, :s, :s]
        y_test = test_reader.read_field("sol")[-ntest:, ::r, ::r][:, :s, :s]

    print("==== Data ====")
    print(f"grid_size={grid_size}  r={r}  N(subsampled)={s}")
    print(f"ntrain={ntrain}  ntest={ntest}")
    print(f"x_train: {tuple(x_train.shape)}  y_train: {tuple(y_train.shape)}")

    # ------------------------------------------------------------------
    # Build + train RFM
    # ------------------------------------------------------------------
    batch_size = int(args.batch_size)
    feature_batch_size = int(args.feature_batch_size)
    feature_batch_size_test = int(args.feature_batch_size_test)

    if device.type == "cuda":
        try:
            free_mem, _ = torch.cuda.mem_get_info(device)
        except (AttributeError, RuntimeError):
            free_mem = None

        if free_mem is not None:
            bytes_per = 4  # float32
            safety_factor = 12
            target = int(free_mem * 0.6)
            orig = (batch_size, feature_batch_size, feature_batch_size_test)
            while True:
                est = batch_size * feature_batch_size * s * s * bytes_per * safety_factor
                if est <= target or (batch_size == 1 and feature_batch_size == 1):
                    break
                if feature_batch_size > 1:
                    feature_batch_size = max(1, feature_batch_size // 2)
                elif batch_size > 1:
                    batch_size = max(1, batch_size // 2)
                else:
                    break
            if feature_batch_size_test > feature_batch_size:
                feature_batch_size_test = max(1, min(feature_batch_size_test, feature_batch_size * 2))
            if (batch_size, feature_batch_size, feature_batch_size_test) != orig:
                print(
                    "Adjusted batch sizes to reduce CUDA memory usage: "
                    f"batch_size={batch_size}, feature_batch_size={feature_batch_size}, "
                    f"feature_batch_size_test={feature_batch_size_test}"
                )

    cfg = RFM2DConfig(
        N=s,
        m=int(args.m),
        lamreg=float(args.lamreg),
        tau_g=float(args.tau_g),
        al_g=float(args.al_g),
        s_plus=float(args.s_plus),
        s_minus=float(args.s_minus),
        delta_sig=float(args.delta_sig),
        smooth_dt=float(args.smooth_dt),
        smooth_eta=float(args.smooth_eta),
        smooth_steps=int(args.smooth_steps),
        batch_size=batch_size,
        feature_batch_size=feature_batch_size,
        feature_batch_size_test=feature_batch_size_test,
        device=device,
        dtype=dtype,
    )
    model = PredictorCorrectorRFM2D(cfg)

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
    plot_error_histogram(test_err.tolist(), os.path.join(args.viz_dir, "error_hist"), title="RFM 2D test error")

    ns = min(int(args.viz_num_samples), ntest)
    for i in range(ns):
        a_i = x_test[i]
        y_i = y_test[i]
        pred_i = model.predict(a_i).detach().cpu().numpy()
        plot_2d_comparison(
            input_field=a_i.detach().cpu().numpy(),
            gt=y_i.detach().cpu().numpy(),
            pred=pred_i,
            out_path_no_ext=os.path.join(args.viz_dir, f"sample_{i:03d}"),
            title_prefix=f"RFM 2D | sample {i}",
        )

    summary_path = os.path.join(args.viz_dir, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("RFM 2D Darcy\n")
        f.write(str(cfg) + "\n")
        f.write(f"train_mean_relL2 {train_err.mean():.8f}\n")
        f.write(f"test_mean_relL2  {test_err.mean():.8f}\n")
        f.write(f"train_seconds    {t_train:.3f}\n")
    print(f"Saved visualizations and summary to: {args.viz_dir}")


if __name__ == "__main__":
    main()
