"""Hyperparameter tuning for 2D RFM (Darcy).

Example:
  python tune_rfm_2d.py --data-mode single_split --data-file data/darcy_data.mat \
    --ntrain 1000 --ntest 100 --r 2 --grid-size 421 --batch-size 2 --m 350 \
    --val-split 0.2 --max-trials 30 --rf-seeds 0 --device cpu \
    --refit-best --save-results results/hpo_rfm_2d.json
"""

from __future__ import annotations

import argparse
import itertools
import os
from timeit import default_timer
from typing import Any

import numpy as np
import torch

from cli_utils import add_data_mode_args, validate_data_mode_args
from hpo_utils import TrialResult, eval_rel_l2, make_loader, parse_csv_list, sample_config, save_results, split_indices
from rfm_core import fit_rfm, save_rfm_model
from rfm_features import DarcyRFFeatures, grf_sample_2d
from utilities3 import MatReader


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="HPO for RFM 2D (Darcy)")
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
    parser.add_argument("--dt", type=float, default=0.03, help="Heat smoothing dt.")
    parser.add_argument("--heat-steps", type=int, default=34, help="Heat smoothing steps.")
    parser.add_argument("--f-const", type=float, default=1.0, help="Darcy forcing constant.")
    parser.add_argument("--feature-chunk", type=int, default=32, help="Feature chunk size for Poisson solves.")
    parser.add_argument("--device", type=str, default=None, help="Device (e.g. cuda or cpu).")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for shuffling.")
    parser.add_argument("--val-split", type=float, default=0.2, help="Fraction of train data used for validation.")
    parser.add_argument("--max-trials", type=int, default=30, help="Number of HPO trials.")
    parser.add_argument("--search", choices=("random", "grid"), default="random", help="Search strategy.")
    parser.add_argument("--rf-seeds", type=str, default="0", help="Random feature seeds (comma-separated).")
    parser.add_argument("--save-results", type=str, default="results/hpo_rfm_2d.json", help="Output path.")
    parser.add_argument("--refit-best", action="store_true", help="Refit best config on train+val.")
    parser.add_argument("--save-best-model", action="store_true", help="Save best model after refit.")
    parser.add_argument("--model-out", type=str, default="model/rfm_2d_best.pt", help="Model output path.")
    parser.add_argument("--lam-values", type=str, default="1e-12,1e-10,1e-8,1e-6,1e-4", help="Lambda choices.")
    parser.add_argument("--tau-theta-range", type=str, default="1.0,30.0", help="Tau log-uniform range.")
    parser.add_argument("--alpha-theta-range", type=str, default="1.0,5.0", help="Alpha uniform range.")
    parser.add_argument("--delta-sig-range", type=str, default="0.03,0.5", help="Delta-sig log-uniform range.")
    parser.add_argument("--s-plus-range", type=str, default="0.01,0.5", help="s_plus log-uniform range.")
    parser.add_argument("--s-minus-range", type=str, default="0.01,0.5", help="s_minus log-uniform range.")
    parser.add_argument("--eta-range", type=str, default="1e-6,1e-3", help="Eta log-uniform range.")
    return parser


def _validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    validate_data_mode_args(args, parser)
    if not 0 < args.val_split < 1:
        parser.error("--val-split must be in (0, 1)")


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


def _parse_range(value: str) -> tuple[float, float]:
    low, high = [float(item.strip()) for item in value.split(",")]
    return low, high


def _build_search_space(args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    tau_low, tau_high = _parse_range(args.tau_theta_range)
    alpha_low, alpha_high = _parse_range(args.alpha_theta_range)
    delta_low, delta_high = _parse_range(args.delta_sig_range)
    s_plus_low, s_plus_high = _parse_range(args.s_plus_range)
    s_minus_low, s_minus_high = _parse_range(args.s_minus_range)
    eta_low, eta_high = _parse_range(args.eta_range)
    lam_values = [float(item) for item in parse_csv_list(args.lam_values, float)]

    return {
        "lam": {"type": "choice", "choices": lam_values},
        "tau_theta": {"type": "loguniform", "low": tau_low, "high": tau_high},
        "alpha_theta": {"type": "uniform", "low": alpha_low, "high": alpha_high},
        "delta_sig": {"type": "loguniform", "low": delta_low, "high": delta_high},
        "s_plus": {"type": "loguniform", "low": s_plus_low, "high": s_plus_high},
        "s_minus": {"type": "loguniform", "low": s_minus_low, "high": s_minus_high},
        "eta": {"type": "loguniform", "low": eta_low, "high": eta_high},
    }


def _grid_configs(space: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    keys = list(space.keys())
    values = []
    for key in keys:
        spec = space[key]
        if spec.get("type") != "choice":
            raise ValueError("Grid search requires all parameters to be of type 'choice'.")
        values.append(spec["choices"])
    configs = []
    for combo in itertools.product(*values):
        configs.append(dict(zip(keys, combo)))
    return configs


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    _validate_args(args, parser)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = _select_device(args.device)
    rf_seeds = [int(seed) for seed in parse_csv_list(args.rf_seeds, int)]

    x_train, y_train, x_test, y_test = _load_data(args)
    train_idx, val_idx = split_indices(x_train.shape[0], args.val_split, args.seed, shuffle=True)
    x_tr, y_tr = x_train[train_idx], y_train[train_idx]
    x_val, y_val = x_train[val_idx], y_train[val_idx]

    train_loader = make_loader(x_tr, y_tr, args.batch_size, shuffle=True)
    val_loader = make_loader(x_val, y_val, args.batch_size, shuffle=False)
    test_loader = make_loader(x_test, y_test, args.batch_size, shuffle=False)

    s = x_train.shape[-1]
    space = _build_search_space(args)
    rng = np.random.default_rng(args.seed)

    if args.search == "grid":
        configs = _grid_configs(space)
        max_trials = len(configs)
    else:
        configs = [sample_config(rng, space) for _ in range(args.max_trials)]
        max_trials = args.max_trials

    results: list[TrialResult] = []
    best: TrialResult | None = None

    for trial_idx in range(max_trials):
        config = configs[trial_idx]
        if config["s_plus"] <= 0 or config["s_minus"] <= 0:
            continue
        s_minus = -abs(config["s_minus"])
        if config["s_plus"] <= s_minus:
            continue
        t0 = default_timer()
        val_errors = []
        for rf_seed in rf_seeds:
            generator = torch.Generator(device=device).manual_seed(rf_seed)
            theta1 = grf_sample_2d(
                args.m, s, tau=config["tau_theta"], alpha=config["alpha_theta"], device=device, generator=generator
            )
            theta2 = grf_sample_2d(
                args.m, s, tau=config["tau_theta"], alpha=config["alpha_theta"], device=device, generator=generator
            )
            features = DarcyRFFeatures(
                theta1=theta1,
                theta2=theta2,
                s_plus=config["s_plus"],
                s_minus=s_minus,
                delta_sig=config["delta_sig"],
                eta=config["eta"],
                dt=args.dt,
                heat_steps=args.heat_steps,
                f_const=args.f_const,
                feature_chunk_size=args.feature_chunk,
            )
            alpha = fit_rfm(train_loader, features, args.m, config["lam"], device)
            val_errors.append(eval_rel_l2(alpha, features, val_loader, args.m, device))
        val_mean = float(np.mean(val_errors))
        val_std = float(np.std(val_errors))
        elapsed = default_timer() - t0
        result = TrialResult(
            trial=trial_idx,
            config={**config, "s_minus": s_minus},
            val_mean=val_mean,
            val_std=val_std,
            test_mean=None,
            test_std=None,
            elapsed_s=elapsed,
        )
        results.append(result)
        if best is None or val_mean < best.val_mean:
            best = result
        print(f"[trial {trial_idx}] val rel L2: {val_mean:.6f} ± {val_std:.6f} (time {elapsed:.2f}s)")

    if best is None:
        raise RuntimeError("No trials were executed.")

    print("Best config (val):", best.config)
    print(f"Best val rel L2: {best.val_mean:.6f} ± {best.val_std:.6f}")

    if args.refit_best:
        trainval_loader = make_loader(x_train, y_train, args.batch_size, shuffle=True)
        test_errors = []
        for rf_seed in rf_seeds:
            generator = torch.Generator(device=device).manual_seed(rf_seed)
            theta1 = grf_sample_2d(
                args.m,
                s,
                tau=best.config["tau_theta"],
                alpha=best.config["alpha_theta"],
                device=device,
                generator=generator,
            )
            theta2 = grf_sample_2d(
                args.m,
                s,
                tau=best.config["tau_theta"],
                alpha=best.config["alpha_theta"],
                device=device,
                generator=generator,
            )
            features = DarcyRFFeatures(
                theta1=theta1,
                theta2=theta2,
                s_plus=best.config["s_plus"],
                s_minus=best.config["s_minus"],
                delta_sig=best.config["delta_sig"],
                eta=best.config["eta"],
                dt=args.dt,
                heat_steps=args.heat_steps,
                f_const=args.f_const,
                feature_chunk_size=args.feature_chunk,
            )
            alpha = fit_rfm(trainval_loader, features, args.m, best.config["lam"], device)
            test_errors.append(eval_rel_l2(alpha, features, test_loader, args.m, device))
            if args.save_best_model and rf_seed == rf_seeds[0]:
                os.makedirs(os.path.dirname(args.model_out), exist_ok=True)
                save_rfm_model(
                    args.model_out,
                    alpha=alpha,
                    theta={"theta1": theta1, "theta2": theta2},
                    hyperparams={
                        "m": float(args.m),
                        "lam": float(best.config["lam"]),
                        "tau_theta": float(best.config["tau_theta"]),
                        "alpha_theta": float(best.config["alpha_theta"]),
                        "s_plus": float(best.config["s_plus"]),
                        "s_minus": float(best.config["s_minus"]),
                        "delta_sig": float(best.config["delta_sig"]),
                        "eta": float(best.config["eta"]),
                        "dt": float(args.dt),
                        "heat_steps": float(args.heat_steps),
                        "f_const": float(args.f_const),
                        "feature_chunk_size": float(args.feature_chunk),
                    },
                )
                print(f"Saved best model to {args.model_out}")
        test_mean = float(np.mean(test_errors))
        test_std = float(np.std(test_errors))
        best.test_mean = test_mean
        best.test_std = test_std
        print(f"Best test rel L2: {test_mean:.6f} ± {test_std:.6f}")

    os.makedirs(os.path.dirname(args.save_results), exist_ok=True)
    save_results(args.save_results, results, best)
    print(f"Saved results to {args.save_results}")


if __name__ == "__main__":
    main()
