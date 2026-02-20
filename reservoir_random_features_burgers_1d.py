from __future__ import annotations

import argparse
import json
import os
import time
import warnings
from dataclasses import asdict

import numpy as np
import torch

from cli_utils import add_data_mode_args, add_split_args, validate_data_mode_args
from pol.ridge import fit_ridge_streaming, predict_linear
from pol.reservoir_1d import Reservoir1DSolver
from pol.theme1_random_features_1d import (
    RandomReservoirFeatureMap1D,
    sample_reservoir_configs,
)
from viz_utils import plot_1d_prediction, plot_error_histogram, rel_l2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Theme 1: Random PDE Reservoir Features + Ridge for 1D Burgers"
    )

    add_data_mode_args(
        parser,
        default_data_mode="single_split",
        default_data_file="data/burgers_data_R10.mat",
        default_train_file=None,
        default_test_file=None,
    )
    add_split_args(parser, default_train_split=0.8, default_seed=0)

    parser.add_argument("--ntrain", type=int, default=1000)
    parser.add_argument("--ntest", type=int, default=100)
    parser.add_argument("--sub", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--dry-run", action="store_true")

    parser.add_argument(
        "--reservoir",
        choices=("reaction_diffusion", "ks", "burgers"),
        default="reaction_diffusion",
    )
    parser.add_argument("--Tr", type=float, default=1.0)
    parser.add_argument("--dt", type=float, default=1e-2)
    parser.add_argument("--ks-dt", type=float, default=0.0)
    parser.add_argument("--K", type=int, default=5)
    parser.add_argument("--feature-times", type=str, default="")
    parser.add_argument("--obs", choices=("full", "points"), default="full")
    parser.add_argument("--J", type=int, default=128)
    parser.add_argument(
        "--sensor-mode", choices=("equispaced", "random"), default="equispaced"
    )
    parser.add_argument("--sensor-seed", type=int, default=0)
    parser.add_argument("--input-scale", type=float, default=1.0)
    parser.add_argument("--input-shift", type=float, default=0.0)
    parser.add_argument("--ks-dealias", action="store_true")

    parser.add_argument("--R", type=int, default=8)
    parser.add_argument("--theta-seed", type=int, default=0)

    parser.add_argument("--rd-nu-range", type=float, nargs=2, default=(1e-4, 1e-2))
    parser.add_argument("--rd-alpha-range", type=float, nargs=2, default=(0.5, 1.5))
    parser.add_argument("--rd-beta-range", type=float, nargs=2, default=(0.5, 1.5))
    parser.add_argument(
        "--rd-nu-dist",
        choices=("loguniform", "uniform"),
        default="loguniform",
    )

    parser.add_argument(
        "--res-burgers-nu-range",
        type=float,
        nargs=2,
        default=(1e-3, 2e-1),
    )
    parser.add_argument(
        "--res-burgers-nu-dist",
        choices=("loguniform", "uniform"),
        default="loguniform",
    )

    parser.add_argument("--ks-nl-range", type=float, nargs=2, default=(0.7, 1.3))
    parser.add_argument("--ks-c2-range", type=float, nargs=2, default=(0.7, 1.3))
    parser.add_argument("--ks-c4-range", type=float, nargs=2, default=(0.7, 1.3))

    parser.add_argument("--use-elm", type=int, choices=(0, 1), default=1)
    parser.add_argument(
        "--elm-mode",
        choices=("per_reservoir", "global"),
        default="per_reservoir",
    )
    parser.add_argument("--elm-h-per", type=int, default=128)
    parser.add_argument("--elm-h", type=int, default=2048)
    parser.add_argument("--elm-activation", choices=("tanh", "relu"), default="tanh")
    parser.add_argument("--elm-seed", type=int, default=0)
    parser.add_argument("--elm-weight-scale", type=float, default=0.0)
    parser.add_argument("--elm-bias-scale", type=float, default=1.0)

    parser.add_argument("--ridge-lambda", type=float, default=1e-4)
    parser.add_argument("--ridge-dtype", choices=("float32", "float64"), default="float64")

    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--out-dir", type=str, default="visualizations/theme1_random_features_1d")
    parser.add_argument(
        "--save-model",
        nargs="?",
        const="model.pt",
        default="",
        help="Save model dictionary. Optional path; if omitted uses out-dir/model.pt",
    )

    args = parser.parse_args()

    if not args.dry_run:
        validate_data_mode_args(args, parser)
    if args.ridge_lambda < 0.0:
        parser.error("--ridge-lambda must be non-negative")
    if args.R <= 0:
        parser.error("--R must be positive")
    if args.sub <= 0:
        parser.error("--sub must be positive")
    if args.reservoir == "ks" and args.dt > 1e-3 and args.ks_dt <= 0.0:
        warnings.warn(
            "KS reservoir is sensitive to dt. Consider --ks-dt 5e-4 or smaller.",
            RuntimeWarning,
        )
    return args


def resolve_device(name: str) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device=cuda was requested but CUDA is unavailable")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ridge_dtype_from_name(name: str) -> torch.dtype:
    if name == "float32":
        return torch.float32
    return torch.float64


def make_dry_run_data(ntrain: int, ntest: int, s: int, seed: int) -> tuple[torch.Tensor, ...]:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)

    x_train = torch.randn((ntrain, s), generator=gen) * 0.6
    x_test = torch.randn((ntest, s), generator=gen) * 0.6
    grid = torch.linspace(0.0, 1.0, s)
    weight = torch.sin(2.0 * torch.pi * grid).unsqueeze(0)

    def make_target(x: torch.Tensor) -> torch.Tensor:
        return 0.8 * torch.roll(x, shifts=3, dims=1) + 0.2 * x.pow(2) + 0.1 * weight

    y_train = make_target(x_train)
    y_test = make_target(x_test)
    return x_train, y_train, x_test, y_test


def load_data(args: argparse.Namespace, s: int) -> tuple[torch.Tensor, ...]:
    if args.dry_run:
        return make_dry_run_data(args.ntrain, args.ntest, s, args.seed)
    from utilities3 import MatReader

    if args.data_mode == "single_split":
        reader = MatReader(args.data_file)
        x_data = reader.read_field("a")[:, :: args.sub]
        y_data = reader.read_field("u")[:, :: args.sub]

        total = x_data.shape[0]
        indices = np.arange(total)
        if args.shuffle:
            rng = np.random.default_rng(args.seed)
            rng.shuffle(indices)

        split_idx = int(total * args.train_split)
        train_idx = indices[:split_idx]
        test_idx = indices[split_idx:]

        if args.ntrain > len(train_idx) or args.ntest > len(test_idx):
            raise ValueError(
                f"Not enough samples for ntrain={args.ntrain}, ntest={args.ntest}. "
                f"train_split={args.train_split}, total={total}."
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
        x_train = train_reader.read_field("a")[: args.ntrain, :: args.sub]
        y_train = train_reader.read_field("u")[: args.ntrain, :: args.sub]
        x_test = test_reader.read_field("a")[: args.ntest, :: args.sub]
        y_test = test_reader.read_field("u")[: args.ntest, :: args.sub]

    return x_train, y_train, x_test, y_test


def build_time_grid(
    *,
    Tr: float,
    dt: float,
    K: int,
    feature_times: str,
) -> tuple[list[float], list[int]]:
    from pol.features_1d import build_time_grid as _build_time_grid

    return _build_time_grid(Tr=Tr, dt=dt, K=K, feature_times=feature_times)


def build_sensor_indices(
    s: int,
    obs: str,
    J: int,
    sensor_mode: str,
    sensor_seed: int,
) -> torch.Tensor:
    from pol.features_1d import build_sensor_indices as _build_sensor_indices

    return _build_sensor_indices(
        s=s,
        obs=obs,
        J=J,
        sensor_mode=sensor_mode,
        sensor_seed=sensor_seed,
    )


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    dt = args.ks_dt if args.reservoir == "ks" and args.ks_dt > 0.0 else args.dt
    s = 2**13 // args.sub

    device = resolve_device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)

    obs_times, obs_steps = build_time_grid(
        Tr=args.Tr,
        dt=dt,
        K=args.K,
        feature_times=args.feature_times,
    )
    sensor_idx = build_sensor_indices(
        s=s,
        obs=args.obs,
        J=args.J,
        sensor_mode=args.sensor_mode,
        sensor_seed=args.sensor_seed,
    )

    theta_configs = sample_reservoir_configs(
        reservoir=args.reservoir,
        R=args.R,
        theta_seed=args.theta_seed,
        rd_nu_range=tuple(args.rd_nu_range),
        rd_alpha_range=tuple(args.rd_alpha_range),
        rd_beta_range=tuple(args.rd_beta_range),
        rd_nu_dist=args.rd_nu_dist,
        res_burgers_nu_range=tuple(args.res_burgers_nu_range),
        res_burgers_nu_dist=args.res_burgers_nu_dist,
        ks_nl_range=tuple(args.ks_nl_range),
        ks_c2_range=tuple(args.ks_c2_range),
        ks_c4_range=tuple(args.ks_c4_range),
        ks_dealias=args.ks_dealias,
    )
    solvers = [Reservoir1DSolver(cfg) for cfg in theta_configs]

    feature_map = RandomReservoirFeatureMap1D(
        solvers=solvers,
        obs_steps=obs_steps,
        obs=args.obs,
        sensor_idx=sensor_idx,
        Tr=args.Tr,
        dt=dt,
        input_scale=args.input_scale,
        input_shift=args.input_shift,
        use_elm=(args.use_elm == 1),
        elm_mode=args.elm_mode,
        elm_h_per=args.elm_h_per,
        elm_h=args.elm_h,
        elm_activation=args.elm_activation,
        elm_seed=args.elm_seed,
        elm_weight_scale=args.elm_weight_scale,
        elm_bias_scale=args.elm_bias_scale,
        device=device,
        dtype=torch.float32,
    )

    if args.use_elm == 0 and feature_map.raw_feature_dim > 100000:
        warnings.warn(
            f"Raw feature dim M={feature_map.raw_feature_dim} is large and may cause huge "
            "Gram matrices. Consider --use-elm 1 with --elm-mode per_reservoir.",
            RuntimeWarning,
        )

    x_train, y_train, x_test, y_test = load_data(args, s)
    x_train = x_train.reshape(args.ntrain, s).float()
    y_train = y_train.reshape(args.ntrain, s).float()
    x_test = x_test.reshape(args.ntest, s).float()
    y_test = y_test.reshape(args.ntest, s).float()

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_train, y_train),
        batch_size=args.batch_size,
        shuffle=True,
    )
    eval_train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_train, y_train),
        batch_size=args.batch_size,
        shuffle=False,
    )
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_test, y_test),
        batch_size=args.batch_size,
        shuffle=False,
    )

    @torch.no_grad()
    def feature_fn(x_batch: torch.Tensor) -> torch.Tensor:
        return feature_map(x_batch)

    probe = feature_fn(x_train[: min(2, args.ntrain)])
    if torch.isnan(probe).any():
        raise RuntimeError("NaN detected in features")

    t0 = time.time()
    ridge_dtype = ridge_dtype_from_name(args.ridge_dtype)
    ridge_state = fit_ridge_streaming(
        train_loader,
        feature_fn,
        args.ridge_lambda,
        dtype=ridge_dtype,
        regularize_bias=False,
    )
    W = ridge_state["W"]

    @torch.no_grad()
    def run_eval(loader):
        rels = []
        preds = []
        ys = []
        xs = []
        for xb, yb in loader:
            feat = feature_fn(xb).to(dtype=W.dtype)
            pred = predict_linear(feat, W).to(dtype=torch.float32)
            y_dev = yb.to(pred.device, dtype=pred.dtype)
            num = torch.linalg.norm((pred - y_dev).reshape(pred.shape[0], -1), dim=1)
            den = torch.linalg.norm(y_dev.reshape(y_dev.shape[0], -1), dim=1)
            rels.append((num / (den + 1e-12)).cpu())
            preds.append(pred.cpu())
            ys.append(yb)
            xs.append(xb)

        pred_all = torch.cat(preds, dim=0)
        y_all = torch.cat(ys, dim=0)
        x_all = torch.cat(xs, dim=0)
        rel_all = torch.cat(rels, dim=0)
        return float(rel_all.mean().item()), pred_all, y_all, x_all

    train_rel, _, _, _ = run_eval(eval_train_loader)
    test_rel, pred_test, y_test_all, x_test_all = run_eval(test_loader)
    elapsed = time.time() - t0

    print(
        "reservoir="
        f"{args.reservoir} R={args.R} K={len(obs_steps)} obs={args.obs} "
        f"M={feature_map.raw_feature_dim} H={feature_map.final_feature_dim}"
    )
    print(f"train relL2: {train_rel:.6f}")
    print(f"test  relL2: {test_rel:.6f}")
    print(f"elapsed sec: {elapsed:.3f}")

    try:
        per_sample = [rel_l2(pred_test[i], y_test_all[i]) for i in range(pred_test.shape[0])]
        plot_error_histogram(per_sample, os.path.join(args.out_dir, "test_relL2_hist"))
        x_grid = np.linspace(0.0, 1.0, s)
        sample_ids = []
        for idx in [0, 1, 2]:
            if 0 <= idx < args.ntest:
                sample_ids.append(idx)
        if not sample_ids and args.ntest > 0:
            sample_ids = [0]
        for idx in sample_ids:
            plot_1d_prediction(
                x=x_grid,
                gt=y_test_all[idx],
                pred=pred_test[idx],
                input_u0=x_test_all[idx],
                out_path_no_ext=os.path.join(args.out_dir, f"sample_{idx:03d}"),
                title_prefix=f"sample {idx}: ",
            )
    except Exception as exc:
        print(f"[viz] visualization failed: {exc}")

    if args.save_model:
        save_path = args.save_model
        if save_path == "model.pt":
            save_path = os.path.join(args.out_dir, save_path)

        state: dict[str, object] = {
            "W_out": W.detach().cpu(),
            "theta_list": [asdict(cfg) for cfg in theta_configs],
            "sensor_idx": sensor_idx.cpu(),
            "obs_times": obs_times,
            "obs_steps": obs_steps,
            "config": vars(args),
        }
        if args.use_elm == 1:
            state["elm_mode"] = feature_map.elm_mode
            state["elm_activation"] = args.elm_activation
            if feature_map.elm_mode == "per_reservoir":
                state["elm_weight_list"] = [elm.weight.detach().cpu() for elm in feature_map.elm_list]
                state["elm_bias_list"] = [elm.bias.detach().cpu() for elm in feature_map.elm_list]
            else:
                assert feature_map.elm_global is not None
                state["elm_weight"] = feature_map.elm_global.weight.detach().cpu()
                state["elm_bias"] = feature_map.elm_global.bias.detach().cpu()

        torch.save(state, save_path)
        print(f"saved model: {save_path}")

    with open(os.path.join(args.out_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "args": vars(args),
                "resolved_dt": dt,
                "obs_times": obs_times,
                "obs_steps": obs_steps,
                "raw_feature_dim": feature_map.raw_feature_dim,
                "final_feature_dim": feature_map.final_feature_dim,
                "train_relL2": train_rel,
                "test_relL2": test_rel,
                "elapsed_sec": elapsed,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )


if __name__ == "__main__":
    main()
