"""Hyperparameter tuning for 1D RFM (Burgers).

Example:
  python tune_rfm_1d.py --data-mode single_split --data-file data/burgers_data_R10.mat \
    --ntrain 1000 --ntest 200 --sub 8 --batch-size 20 --m 1024 \
    --val-split 0.2 --max-trials 40 --search random --rf-seeds 0,1,2 --device cuda \
    --refit-best --save-best-model --model-out model/rfm_1d_best.pt \
    --save-results results/hpo_rfm_1d.json
"""

from __future__ import annotations

import argparse
import itertools
import os
from timeit import default_timer
from typing import Any

import numpy as np
import torch

from cli_utils import add_data_mode_args, add_split_args, validate_data_mode_args
from hpo_utils import TrialResult, eval_rel_l2, make_loader, parse_csv_list, sample_config, save_results, split_indices
from rfm_core import fit_rfm, save_rfm_model
from rfm_features import BurgersRFFeatures, grf_sample_1d
from utilities3 import MatReader


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="HPO for RFM 1D (Burgers)")
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
    parser.add_argument("--device", type=str, default=None, help="Device (e.g. cuda or cpu).")
    parser.add_argument("--val-split", type=float, default=0.2, help="Fraction of train data used for validation.")
    parser.add_argument("--max-trials", type=int, default=30, help="Number of HPO trials.")
    parser.add_argument("--search", choices=("random", "grid"), default="random", help="Search strategy.")
    parser.add_argument("--rf-seeds", type=str, default="0", help="Random feature seeds (comma-separated).")
    parser.add_argument("--save-results", type=str, default="results/hpo_rfm_1d.json", help="Output path.")
    parser.add_argument("--refit-best", action="store_true", help="Refit best config on train+val.")
    parser.add_argument("--save-best-model", action="store_true", help="Save best model after refit.")
    parser.add_argument("--model-out", type=str, default="model/rfm_1d_best.pt", help="Model output path.")
    parser.add_argument("--lam-values", type=str, default="0,1e-12,1e-10,1e-8,1e-6,1e-4", help="Lambda choices.")
    parser.add_argument("--delta-range", type=str, default="1e-4,1.0", help="Delta log-uniform range.")
    parser.add_argument("--beta-range", type=str, default="0.05,8.0", help="Beta uniform range.")
    parser.add_argument("--tau-theta-range", type=str, default="1.0,30.0", help="Tau log-uniform range.")
    parser.add_argument("--alpha-theta-range", type=str, default="1.0,5.0", help="Alpha uniform range.")
    add_split_args(parser, default_train_split=0.8, default_seed=0)
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
    sub = args.sub
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


def _parse_range(value: str) -> tuple[float, float]:
    low, high = [float(item.strip()) for item in value.split(",")]
    return low, high


def _build_search_space(args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    delta_low, delta_high = _parse_range(args.delta_range)
    beta_low, beta_high = _parse_range(args.beta_range)
    tau_low, tau_high = _parse_range(args.tau_theta_range)
    alpha_low, alpha_high = _parse_range(args.alpha_theta_range)
    lam_values = [float(item) for item in parse_csv_list(args.lam_values, float)]

    return {
        "lam": {"type": "choice", "choices": lam_values},
        "delta": {"type": "loguniform", "low": delta_low, "high": delta_high},
        "beta": {"type": "uniform", "low": beta_low, "high": beta_high},
        "tau_theta": {"type": "loguniform", "low": tau_low, "high": tau_high},
        "alpha_theta": {"type": "uniform", "low": alpha_low, "high": alpha_high},
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
        t0 = default_timer()
        val_errors = []
        for rf_seed in rf_seeds:
            generator = torch.Generator(device=device).manual_seed(rf_seed)
            theta = grf_sample_1d(
                args.m,
                s,
                tau=config["tau_theta"],
                alpha=config["alpha_theta"],
                device=device,
                generator=generator,
            )
            features = BurgersRFFeatures(theta=theta, delta=config["delta"], beta=config["beta"])
            alpha = fit_rfm(train_loader, features, args.m, config["lam"], device)
            val_errors.append(eval_rel_l2(alpha, features, val_loader, args.m, device))
        val_mean = float(np.mean(val_errors))
        val_std = float(np.std(val_errors))
        elapsed = default_timer() - t0
        result = TrialResult(
            trial=trial_idx,
            config=config,
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
            theta = grf_sample_1d(
                args.m,
                s,
                tau=best.config["tau_theta"],
                alpha=best.config["alpha_theta"],
                device=device,
                generator=generator,
            )
            features = BurgersRFFeatures(
                theta=theta, delta=best.config["delta"], beta=best.config["beta"]
            )
            alpha = fit_rfm(trainval_loader, features, args.m, best.config["lam"], device)
            test_errors.append(eval_rel_l2(alpha, features, test_loader, args.m, device))
            if args.save_best_model and rf_seed == rf_seeds[0]:
                os.makedirs(os.path.dirname(args.model_out), exist_ok=True)
                save_rfm_model(
                    args.model_out,
                    alpha=alpha,
                    theta={"theta": theta},
                    hyperparams={
                        "m": float(args.m),
                        "lam": float(best.config["lam"]),
                        "delta": float(best.config["delta"]),
                        "beta": float(best.config["beta"]),
                        "tau_theta": float(best.config["tau_theta"]),
                        "alpha_theta": float(best.config["alpha_theta"]),
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
