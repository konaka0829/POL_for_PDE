from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
import warnings

import numpy as np
import torch

from cli_utils import add_data_mode_args, add_split_args, validate_data_mode_args
from pol.model123_1d import Model1Predictor1D, Model2Regressor1D, Model3Regressor1D, Model123Config
from viz_utils import plot_1d_prediction, plot_error_histogram


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Model 1 / 2 / 3 runner for 1D Burgers target")
    parser.add_argument("--model", choices=("model1", "model2", "model3"), required=True)
    parser.add_argument(
        "--reservoir",
        choices=("burgers", "reaction_diffusion", "ks"),
        default="burgers",
    )

    add_data_mode_args(
        parser,
        default_data_mode="single_split",
        default_data_file="data/burgers_model123.mat",
        default_train_file=None,
        default_test_file=None,
    )
    add_split_args(parser, default_train_split=0.75, default_seed=0)
    parser.add_argument("--ntrain", type=int, default=384)
    parser.add_argument("--ntest", type=int, default=128)
    parser.add_argument("--sub", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)

    parser.add_argument("--T", type=float, default=1.0)
    parser.add_argument("--Ttilde", type=float, default=0.0)
    parser.add_argument("--dt", type=float, default=1e-2)
    parser.add_argument("--feature-times", type=str, default="")
    parser.add_argument("--K", type=int, default=1)

    parser.add_argument("--obs", choices=("full", "points", "fourier", "proj"), default="full")
    parser.add_argument("--J", type=int, default=128)
    parser.add_argument("--sensor-mode", choices=("equispaced", "random"), default="equispaced")
    parser.add_argument("--sensor-seed", type=int, default=0)

    parser.add_argument("--input-scale", type=float, default=1.0)
    parser.add_argument("--input-shift", type=float, default=0.0)

    parser.add_argument("--ridge-lambda", type=float, default=1e-4)
    parser.add_argument("--ridge-dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--standardize-features", type=int, choices=(0, 1), default=0)
    parser.add_argument("--feature-std-eps", type=float, default=1e-6)

    parser.add_argument("--elm-h", type=int, default=1024)
    parser.add_argument("--elm-activation", choices=("tanh", "relu", "identity"), default="tanh")
    parser.add_argument("--elm-seed", type=int, default=0)
    parser.add_argument("--elm-weight-scale", type=float, default=0.0)
    parser.add_argument("--elm-bias-scale", type=float, default=1.0)

    parser.add_argument("--rd-nu", type=float, default=1e-3)
    parser.add_argument("--rd-alpha", type=float, default=1.0)
    parser.add_argument("--rd-beta", type=float, default=1.0)
    parser.add_argument("--res-burgers-nu", type=float, default=0.05)
    parser.add_argument("--res-burgers-b", type=float, default=1.0)
    parser.add_argument("--ks-dealias", action="store_true")
    parser.add_argument("--ks-b", type=float, default=1.0)
    parser.add_argument("--ks-eta", type=float, default=1.0)
    parser.add_argument("--ks-kappa", type=float, default=1.0)
    parser.add_argument("--burgers-scheme", choices=("semi_implicit", "split_step"), default="split_step")
    parser.add_argument("--burgers-fine-dt", type=float, default=1e-4)
    parser.add_argument("--burgers-dealias", type=int, choices=(0, 1), default=1)

    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--out-dir", type=str, default="outputs/model123_burgers_1d")
    parser.add_argument(
        "--save-model",
        nargs="?",
        const="model.pt",
        default="",
        help="Optional save path. If omitted, writes out-dir/model.pt",
    )
    args = parser.parse_args()
    validate_data_mode_args(args, parser)
    if args.Ttilde <= 0.0:
        args.Ttilde = args.T
    if args.T <= 0.0 or args.Ttilde <= 0.0 or args.dt <= 0.0:
        parser.error("--T, --Ttilde, and --dt must be positive")
    if args.ridge_lambda < 0.0:
        parser.error("--ridge-lambda must be non-negative")
    if args.feature_std_eps <= 0.0:
        parser.error("--feature-std-eps must be positive")
    if args.sub <= 0 or args.batch_size <= 0 or args.ntrain <= 0 or args.ntest <= 0:
        parser.error("--sub, --batch-size, --ntrain, and --ntest must be positive")
    return args


def ridge_dtype_from_name(name: str) -> torch.dtype:
    return torch.float32 if name == "float32" else torch.float64


def _extract_scalar_meta(reader, field: str) -> float | None:
    if field not in reader.data:
        return None
    value = reader.read_field(field)
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return None
        return float(value.reshape(-1)[0].item())
    arr = np.asarray(value)
    if arr.size == 0:
        return None
    return float(arr.reshape(-1)[0])


def _validate_shapes(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
) -> None:
    if x_train.shape[1] != y_train.shape[1]:
        raise ValueError(
            f"Train a/u resolution mismatch: {x_train.shape[1]} vs {y_train.shape[1]}"
        )
    if x_test.shape[1] != y_test.shape[1]:
        raise ValueError(
            f"Test a/u resolution mismatch: {x_test.shape[1]} vs {y_test.shape[1]}"
        )
    if x_train.shape[1] != x_test.shape[1]:
        raise ValueError(
            f"Train/test input resolution mismatch: {x_train.shape[1]} vs {x_test.shape[1]}"
        )
    if y_train.shape[1] != y_test.shape[1]:
        raise ValueError(
            f"Train/test output resolution mismatch: {y_train.shape[1]} vs {y_test.shape[1]}"
        )


def _validate_target_time(args: argparse.Namespace, train_reader, test_reader=None) -> None:
    train_T = _extract_scalar_meta(train_reader, "T")
    test_T = _extract_scalar_meta(test_reader, "T") if test_reader is not None else train_T
    available = [val for val in (train_T, test_T) if val is not None]
    if not available:
        warnings.warn(
            "Dataset does not provide T metadata; --T cannot be validated against the target file.",
            RuntimeWarning,
        )
        return
    if train_T is not None and test_T is not None and not np.isclose(train_T, test_T):
        raise ValueError(f"Train/test target-time metadata mismatch: {train_T} vs {test_T}")
    data_T = available[0]
    if not np.isclose(data_T, args.T):
        raise ValueError(
            f"Requested --T={args.T} but dataset target time is T={data_T}. "
            "The CLI target time must match the dataset target."
        )


def load_data(args: argparse.Namespace) -> tuple[torch.Tensor, ...]:
    from utilities3 import MatReader

    if args.data_mode == "single_split":
        reader = MatReader(args.data_file)
        _validate_target_time(args, reader)
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
                f"Not enough samples for ntrain={args.ntrain}, ntest={args.ntest}, total={total}"
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
        _validate_target_time(args, train_reader, test_reader)
        x_train = train_reader.read_field("a")[: args.ntrain, :: args.sub]
        y_train = train_reader.read_field("u")[: args.ntrain, :: args.sub]
        x_test = test_reader.read_field("a")[: args.ntest, :: args.sub]
        y_test = test_reader.read_field("u")[: args.ntest, :: args.sub]
    _validate_shapes(x_train, y_train, x_test, y_test)
    s = int(x_train.shape[1])
    return (
        x_train.reshape(args.ntrain, s).float(),
        y_train.reshape(args.ntrain, s).float(),
        x_test.reshape(args.ntest, s).float(),
        y_test.reshape(args.ntest, s).float(),
    )


def build_model_config(args: argparse.Namespace) -> Model123Config:
    return Model123Config(
        reservoir=args.reservoir,
        Ttilde=args.Ttilde,
        dt=args.dt,
        K=args.K,
        feature_times=args.feature_times,
        obs=args.obs,
        J=args.J,
        sensor_mode=args.sensor_mode,
        sensor_seed=args.sensor_seed,
        input_scale=args.input_scale,
        input_shift=args.input_shift,
        ridge_lambda=args.ridge_lambda,
        ridge_dtype=ridge_dtype_from_name(args.ridge_dtype),
        standardize_features=bool(args.standardize_features),
        feature_std_eps=args.feature_std_eps,
        elm_hidden_dim=args.elm_h,
        elm_activation=args.elm_activation,
        elm_seed=args.elm_seed,
        elm_weight_scale=args.elm_weight_scale,
        elm_bias_scale=args.elm_bias_scale,
        rd_nu=args.rd_nu,
        rd_alpha=args.rd_alpha,
        rd_beta=args.rd_beta,
        res_burgers_nu=args.res_burgers_nu,
        res_burgers_b=args.res_burgers_b,
        ks_dealias=args.ks_dealias,
        ks_b=args.ks_b,
        ks_eta=args.ks_eta,
        ks_kappa=args.ks_kappa,
        burgers_scheme=args.burgers_scheme,
        burgers_fine_dt=args.burgers_fine_dt,
        burgers_dealias=bool(args.burgers_dealias),
        device=args.device,
        dtype=torch.float32,
    )


@torch.no_grad()
def evaluate_model(model, loader):
    rels = []
    preds = []
    ys = []
    xs = []
    for xb, yb in loader:
        pred = model.predict(xb).cpu()
        num = torch.linalg.norm((pred - yb).reshape(pred.shape[0], -1), dim=1)
        den = torch.linalg.norm(yb.reshape(yb.shape[0], -1), dim=1)
        rels.append((num / (den + 1e-12)).cpu())
        preds.append(pred)
        ys.append(yb)
        xs.append(xb)
    return float(torch.cat(rels).mean().item()), torch.cat(preds), torch.cat(ys), torch.cat(xs)


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    x_train, y_train, x_test, y_test = load_data(args)
    s = int(x_train.shape[1])
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

    model_cfg = build_model_config(args)
    if args.model == "model1":
        model = Model1Predictor1D(s=s, config=model_cfg)
        ridge_state = None
    elif args.model == "model2":
        model = Model2Regressor1D(s=s, config=model_cfg)
        ridge_state = model.fit(train_loader)
    else:
        model = Model3Regressor1D(s=s, config=model_cfg)
        ridge_state = model.fit(train_loader)

    train_rel, _, _, _ = evaluate_model(model, eval_train_loader)
    test_rel, pred_test, y_test_all, x_test_all = evaluate_model(model, test_loader)

    resolved_cfg = getattr(model, "config", model_cfg)
    actual_obs = resolved_cfg.obs
    actual_J = resolved_cfg.J
    print(f"model={args.model} reservoir={args.reservoir} obs={actual_obs} J={actual_J}")
    print(f"T={args.T} Ttilde={args.Ttilde} dt={args.dt}")
    print(f"train relL2: {train_rel:.6f}")
    print(f"test  relL2: {test_rel:.6f}")

    per_sample = []
    for idx in range(pred_test.shape[0]):
        num = torch.linalg.norm((pred_test[idx] - y_test_all[idx]).reshape(-1))
        den = torch.linalg.norm(y_test_all[idx].reshape(-1))
        per_sample.append(float((num / (den + 1e-12)).item()))
    plot_error_histogram(per_sample, os.path.join(args.out_dir, "test_relL2_hist"))

    x_grid = np.linspace(0.0, 1.0, s, endpoint=False)
    for idx in [0, min(1, args.ntest - 1), min(2, args.ntest - 1)]:
        plot_1d_prediction(
            x=x_grid,
            gt=y_test_all[idx],
            pred=pred_test[idx],
            input_u0=x_test_all[idx],
            out_path_no_ext=os.path.join(args.out_dir, f"sample_{idx:03d}"),
            title_prefix=f"{args.model} sample {idx}: ",
        )

    if args.save_model:
        save_path = args.save_model
        if save_path == "model.pt":
            save_path = os.path.join(args.out_dir, save_path)
        state = {
            "args": vars(args),
            "model_config": asdict(resolved_cfg),
        }
        if ridge_state is not None:
            for key, value in ridge_state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.detach().cpu()
        if hasattr(model, "weight") and getattr(model, "weight", None) is not None:
            state["W_out"] = model.weight.detach().cpu()
        if hasattr(model, "elm") and getattr(model, "elm", None) is not None:
            state["elm_weight"] = model.elm.weight.detach().cpu()
            state["elm_bias"] = model.elm.bias.detach().cpu()
            state["elm_activation"] = model.elm.activation
        torch.save(state, save_path)
        print(f"saved model: {save_path}")

    with open(os.path.join(args.out_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "args": vars(args),
                "resolved_obs": actual_obs,
                "resolved_J": actual_J,
                "train_relL2": train_rel,
                "test_relL2": test_rel,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )


if __name__ == "__main__":
    main()
