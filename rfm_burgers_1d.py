from __future__ import annotations

import argparse
import json
import math
import os
import time
import warnings
from dataclasses import asdict

import numpy as np
import torch

from cli_utils import add_data_mode_args, add_split_args, validate_data_mode_args
from pol.encoder_1d import FixedEncoder1D
from pol.features_1d import build_time_grid
from pol.reservoir_1d import Reservoir1DSolver, ReservoirConfig
from viz_utils import plot_1d_prediction, plot_error_histogram, rel_l2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backprop-free function-valued RFM for 1D Burgers"
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

    parser.add_argument("--rd-nu", type=float, default=1e-3)
    parser.add_argument("--rd-alpha", type=float, default=1.0)
    parser.add_argument("--rd-beta", type=float, default=1.0)
    parser.add_argument("--res-burgers-nu", type=float, default=5e-2)
    parser.add_argument("--ks-dealias", action="store_true")

    parser.add_argument("--input-scale", type=float, default=1.0)
    parser.add_argument("--input-shift", type=float, default=0.0)
    parser.add_argument(
        "--encoder",
        choices=("linear", "fourier_filter", "randconv", "fourier_rfm", "poly_deriv"),
        default="linear",
    )
    parser.add_argument("--encoder-center", type=int, choices=(0, 1), default=0)
    parser.add_argument("--encoder-standardize", type=int, choices=(0, 1), default=0)
    parser.add_argument("--encoder-standardize-eps", type=float, default=1e-6)
    parser.add_argument("--encoder-post", choices=("none", "tanh", "clip"), default="none")
    parser.add_argument("--encoder-tanh-gamma", type=float, default=1.0)
    parser.add_argument("--encoder-clip-c", type=float, default=3.0)

    parser.add_argument(
        "--encoder-fourier-mode",
        choices=("lowpass", "bandpass", "randphase", "randamp", "randcomplex"),
        default="lowpass",
    )
    parser.add_argument("--encoder-fourier-kmin", type=int, default=0)
    parser.add_argument("--encoder-fourier-kmax", type=int, default=16)
    parser.add_argument("--encoder-fourier-seed", type=int, default=0)
    parser.add_argument("--encoder-fourier-amp-std", type=float, default=0.5)
    parser.add_argument("--encoder-fourier-output-scale", type=float, default=1.0)

    parser.add_argument("--encoder-randconv-kernel-size", type=int, default=33)
    parser.add_argument("--encoder-randconv-seed", type=int, default=0)
    parser.add_argument("--encoder-randconv-std", type=float, default=1.0)
    parser.add_argument("--encoder-randconv-normalize", choices=("none", "l1", "l2"), default="l2")

    parser.add_argument("--encoder-fourier-rfm-C", type=int, default=1)
    parser.add_argument("--encoder-fourier-rfm-mode", choices=("sum", "mean", "ensemble"), default="sum")
    parser.add_argument(
        "--encoder-fourier-rfm-activation",
        choices=("tanh", "relu", "identity"),
        default="tanh",
    )
    parser.add_argument("--encoder-fourier-rfm-kmin", type=int, default=0)
    parser.add_argument("--encoder-fourier-rfm-kmax", type=int, default=16)
    parser.add_argument("--encoder-fourier-rfm-seed", type=int, default=0)
    parser.add_argument("--encoder-fourier-rfm-theta-scale", type=float, default=1.0)
    parser.add_argument("--encoder-fourier-rfm-output-scale", type=float, default=1.0)

    parser.add_argument("--encoder-poly-a1", type=float, default=1.0)
    parser.add_argument("--encoder-poly-a2", type=float, default=0.0)
    parser.add_argument("--encoder-poly-a3", type=float, default=0.0)

    parser.add_argument("--forcing-mode", choices=("none", "constant", "window"), default="none")
    parser.add_argument("--forcing-gamma", type=float, default=0.0)
    parser.add_argument("--forcing-source", choices=("raw", "pre", "z0"), default="pre")
    parser.add_argument("--forcing-tstart", type=float, default=0.0)
    parser.add_argument("--forcing-tend", type=float, default=0.0)

    parser.add_argument("--m", type=int, default=256)
    parser.add_argument(
        "--rfm-activation", choices=("tanh", "relu", "identity"), default="tanh"
    )
    parser.add_argument("--rfm-seed", type=int, default=0)
    parser.add_argument("--rfm-weight-scale", type=float, default=0.0)
    parser.add_argument("--rfm-bias-scale", type=float, default=1.0)

    parser.add_argument("--ridge-lambda", type=float, default=1e-4)
    parser.add_argument("--ridge-dtype", choices=("float32", "float64"), default="float64")

    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--out-dir", type=str, default="visualizations/rfm_burgers_1d")
    parser.add_argument(
        "--save-model",
        nargs="?",
        const="model.pt",
        default="",
        help="Save model dictionary. Optional path; if omitted uses out-dir/model.pt",
    )

    parser.add_argument("--dry-run", action="store_true")

    args = parser.parse_args()
    if not args.dry_run:
        validate_data_mode_args(args, parser)

    if args.reservoir == "ks" and args.dt > 1e-3 and args.ks_dt <= 0.0:
        warnings.warn(
            "KS reservoir is sensitive to dt. Consider --ks-dt 5e-4 or smaller.",
            RuntimeWarning,
        )

    if args.ridge_lambda < 0.0:
        parser.error("--ridge-lambda must be non-negative")
    if args.m <= 0:
        parser.error("--m must be positive")
    if args.encoder_standardize_eps <= 0.0:
        parser.error("--encoder-standardize-eps must be positive")
    if args.encoder_clip_c <= 0.0:
        parser.error("--encoder-clip-c must be positive")
    if args.encoder_fourier_rfm_C <= 0:
        parser.error("--encoder-fourier-rfm-C must be positive")
    if args.forcing_mode == "window":
        if args.forcing_tstart > args.forcing_tend:
            parser.error("--forcing-tstart must be <= --forcing-tend")
        if args.forcing_tstart < 0.0:
            parser.error("--forcing-tstart must be >= 0")
        if args.forcing_tend < 0.0:
            parser.error("--forcing-tend must be >= 0")
    if args.forcing_mode != "none" and args.forcing_gamma == 0.0:
        warnings.warn(
            "forcing-mode is enabled but forcing-gamma is zero; forcing becomes inactive.",
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


def resolve_forcing_steps(
    *,
    forcing_mode: str,
    forcing_tstart: float,
    forcing_tend: float,
    dt: float,
    Tr: float,
    max_obs_step: int,
) -> tuple[int, int] | None:
    if forcing_mode != "window":
        return None
    if forcing_tstart > forcing_tend:
        raise ValueError("forcing_tstart must be <= forcing_tend")
    if forcing_tstart < 0.0 or forcing_tend > Tr:
        raise ValueError("forcing window must satisfy 0 <= tstart <= tend <= Tr")
    start_step = max(1, int(round(forcing_tstart / dt)))
    end_step = min(max_obs_step, int(round(forcing_tend / dt)))
    return (start_step, end_step)


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


def apply_activation(x: torch.Tensor, name: str) -> torch.Tensor:
    if name == "tanh":
        return torch.tanh(x)
    if name == "relu":
        return torch.relu(x)
    return x


def main() -> None:
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    dt = args.ks_dt if args.reservoir == "ks" and args.ks_dt > 0.0 else args.dt
    s = 2**13 // args.sub

    device = resolve_device(args.device)
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    times, obs_steps = build_time_grid(Tr=args.Tr, dt=dt, K=args.K, feature_times=args.feature_times)
    k_obs = len(obs_steps)
    max_obs_step = obs_steps[-1]
    if args.forcing_mode == "window":
        if args.forcing_tstart > args.forcing_tend:
            raise ValueError("forcing_tstart must be <= forcing_tend")
        if args.forcing_tstart < 0.0 or args.forcing_tend > args.Tr:
            raise ValueError("forcing window must satisfy 0 <= tstart <= tend <= Tr")

    x_train, y_train, x_test, y_test = load_data(args, s)
    x_train = x_train.reshape(args.ntrain, s).float()
    y_train = y_train.reshape(args.ntrain, s).float()
    x_test = x_test.reshape(args.ntest, s).float()
    y_test = y_test.reshape(args.ntest, s).float()

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_train, y_train),
        batch_size=args.batch_size,
        shuffle=False,
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

    reservoir = Reservoir1DSolver(
        ReservoirConfig(
            reservoir=args.reservoir,
            rd_nu=args.rd_nu,
            rd_alpha=args.rd_alpha,
            rd_beta=args.rd_beta,
            res_burgers_nu=args.res_burgers_nu,
            ks_dealias=args.ks_dealias,
        )
    )
    encoder = FixedEncoder1D(s=s, device=device, dtype=torch.float32, args=args)
    forcing_steps = resolve_forcing_steps(
        forcing_mode=args.forcing_mode,
        forcing_tstart=args.forcing_tstart,
        forcing_tend=args.forcing_tend,
        dt=dt,
        Tr=args.Tr,
        max_obs_step=max_obs_step,
    )

    num_members = 1
    if args.encoder == "fourier_rfm" and args.encoder_fourier_rfm_mode == "ensemble":
        num_members = args.encoder_fourier_rfm_C
    k_obs_total = k_obs * num_members

    gen = torch.Generator(device="cpu")
    gen.manual_seed(args.rfm_seed)
    weight_scale = args.rfm_weight_scale
    if weight_scale <= 0.0:
        weight_scale = 1.0 / math.sqrt(float(k_obs_total))
    A = weight_scale * torch.randn((args.m, k_obs_total), generator=gen, dtype=torch.float32, device="cpu")
    b = args.rfm_bias_scale * torch.randn((args.m,), generator=gen, dtype=torch.float32, device="cpu")
    A = A.to(device)
    b = b.to(device)

    @torch.no_grad()
    def function_features(x_batch: torch.Tensor) -> torch.Tensor:
        enc = encoder.encode(x_batch.to(device=device, dtype=torch.float32))
        forcing_active = args.forcing_mode != "none" and args.forcing_gamma != 0.0

        z_members = []
        for i, z0_member in enumerate(enc.z0_list):
            forcing_member = None
            if forcing_active:
                if args.forcing_source == "raw":
                    source = enc.x_raw
                elif args.forcing_source == "pre":
                    source = enc.x_pre_list[i]
                elif args.forcing_source == "z0":
                    source = enc.z0_list[i]
                else:
                    raise ValueError(f"Unsupported forcing source: {args.forcing_source}")
                forcing_member = args.forcing_gamma * source

            states = reservoir.simulate(
                z0_member,
                dt=dt,
                Tr=args.Tr,
                obs_steps=obs_steps,
                forcing=forcing_member,
                forcing_steps=forcing_steps,
            )
            z_members.append(torch.stack(states, dim=1))

        if len(z_members) == 1:
            Z_total = z_members[0]
        else:
            Z_total = torch.cat(z_members, dim=1)

        mixed = torch.einsum("bks,mk->bms", Z_total, A) + b.view(1, args.m, 1)
        feat = apply_activation(mixed, args.rfm_activation)
        return torch.nan_to_num(feat, nan=0.0, posinf=1e6, neginf=-1e6)

    probe = function_features(x_train[: min(2, args.ntrain)])
    if probe.shape[-2:] != (args.m, s):
        raise RuntimeError(f"Unexpected feature shape: {tuple(probe.shape)}")
    if torch.isnan(probe).any():
        raise RuntimeError("NaN detected in RFM features")

    ridge_dtype = ridge_dtype_from_name(args.ridge_dtype)
    t0 = time.time()
    gram = torch.zeros((args.m, args.m), dtype=ridge_dtype, device=device)
    rhs = torch.zeros((args.m,), dtype=ridge_dtype, device=device)

    for xb, yb in train_loader:
        F = function_features(xb).to(dtype=ridge_dtype)
        yb_dev = yb.to(device=device, dtype=ridge_dtype)
        gram += torch.einsum("bms,bns->mn", F, F)
        rhs += torch.einsum("bms,bs->m", F, yb_dev)

    eye = torch.eye(args.m, device=device, dtype=ridge_dtype)
    reg_gram = gram + args.ridge_lambda * eye
    chol = torch.linalg.cholesky(reg_gram)
    alpha = torch.cholesky_solve(rhs.unsqueeze(1), chol).squeeze(1)

    @torch.no_grad()
    def run_eval(loader):
        rels = []
        preds = []
        ys = []
        xs = []
        for xb, yb in loader:
            F = function_features(xb).to(dtype=alpha.dtype)
            pred = torch.einsum("m,bms->bs", alpha, F).to(dtype=torch.float32)
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

    print(f"reservoir={args.reservoir} m={args.m} act={args.rfm_activation}")
    print(f"dt={dt} Tr={args.Tr} K={k_obs_total} s={s}")
    print(f"train relL2: {train_rel:.6f}")
    print(f"test  relL2: {test_rel:.6f}")
    print(f"elapsed sec: {elapsed:.3f}")

    try:
        per_sample = [rel_l2(pred_test[i], y_test_all[i]) for i in range(pred_test.shape[0])]
        plot_error_histogram(per_sample, os.path.join(out_dir, "test_relL2_hist"))

        x_grid = np.linspace(0.0, 1.0, s)
        sample_ids = [0, min(1, args.ntest - 1), min(2, args.ntest - 1)]
        for idx in sample_ids:
            plot_1d_prediction(
                x=x_grid,
                gt=y_test_all[idx],
                pred=pred_test[idx],
                input_u0=x_test_all[idx],
                out_path_no_ext=os.path.join(out_dir, f"sample_{idx:03d}"),
                title_prefix=f"sample {idx}: ",
            )
    except Exception as exc:
        print(f"[viz] visualization failed: {exc}")

    if args.save_model:
        save_path = args.save_model
        if save_path == "model.pt":
            save_path = os.path.join(out_dir, save_path)
        state = {
            "alpha": alpha.detach().cpu(),
            "A_time_mix": A.detach().cpu(),
            "b_time_mix": b.detach().cpu(),
            "obs_times": times,
            "obs_steps": obs_steps,
            "config": vars(args),
            "reservoir_config": asdict(reservoir.config),
        }
        torch.save(state, save_path)
        print(f"saved model: {save_path}")

    with open(os.path.join(out_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "args": vars(args),
                "resolved_dt": dt,
                "obs_steps": obs_steps,
                "obs_times": times,
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
