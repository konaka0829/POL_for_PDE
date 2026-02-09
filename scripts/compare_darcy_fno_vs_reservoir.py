import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from viz_utils import (
    LearningCurve,
    plot_2d_comparison,
    plot_error_histogram,
    plot_learning_curve,
    rel_l2,
)
from rfno.data_utils import MatReader, UnitGaussianNormalizer
from rfno.darcy_generate import generate_darcy_dataset, save_dataset_to_mat
from rfno.models_fno2d import FNO2d
from rfno.reservoir_readout import RidgeReadout2D


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def parse_lambdas(text: str) -> list[float]:
    vals = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        vals.append(float(item))
    if not vals:
        raise ValueError("--ridge-lambdas must contain at least one value.")
    return vals


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        print("[warn] CUDA is not available. Falling back to CPU.")
        return torch.device("cpu")
    return torch.device(device)


def _ensure_tensor(x: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(x):
        return torch.tensor(x, dtype=torch.float32)
    return x.float()


def load_mat_data(args: argparse.Namespace) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    r = args.r
    s = int(((args.grid_size - 1) / r) + 1)

    if args.data_mode == "single_split":
        reader = MatReader(args.data_file)
        x_data = reader.read_field("coeff")[:, ::r, ::r][:, :s, :s]
        y_data = reader.read_field("sol")[:, ::r, ::r][:, :s, :s]

        x_train = x_data[: args.ntrain]
        y_train = y_data[: args.ntrain]
        x_test = x_data[-args.ntest :]
        y_test = y_data[-args.ntest :]
    else:
        reader = MatReader(args.train_file)
        x_train = reader.read_field("coeff")[: args.ntrain, ::r, ::r][:, :s, :s]
        y_train = reader.read_field("sol")[: args.ntrain, ::r, ::r][:, :s, :s]

        reader.load_file(args.test_file)
        x_test = reader.read_field("coeff")[: args.ntest, ::r, ::r][:, :s, :s]
        y_test = reader.read_field("sol")[: args.ntest, ::r, ::r][:, :s, :s]

    return _ensure_tensor(x_train), _ensure_tensor(y_train), _ensure_tensor(x_test), _ensure_tensor(y_test), s


def load_generate_data(args: argparse.Namespace) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    s = args.gen_grid_size
    n_total = args.ntrain + args.ntest

    coeff, sol = generate_darcy_dataset(
        n_samples=n_total,
        s=s,
        map_type=args.gen_map_type,
        a_pos=args.gen_a_pos,
        a_neg=args.gen_a_neg,
        alpha=args.gen_alpha,
        tau=args.gen_tau,
        forcing=args.gen_forcing,
        grf_device="cpu",
        solver=args.gen_solver,
        cg_tol=args.gen_cg_tol,
        cg_maxiter=args.gen_cg_maxiter,
        seed=args.seed,
    )

    x_train = coeff[: args.ntrain]
    y_train = sol[: args.ntrain]
    x_test = coeff[args.ntrain : args.ntrain + args.ntest]
    y_test = sol[args.ntrain : args.ntrain + args.ntest]

    if args.gen_save_mat:
        save_dataset_to_mat(args.gen_save_mat, coeff, sol)
        print(f"[info] generated dataset saved to {args.gen_save_mat}")

    return x_train, y_train, x_test, y_test, s


def load_data(args: argparse.Namespace) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    if args.data_source == "mat":
        return load_mat_data(args)
    return load_generate_data(args)


@dataclass
class ModelEval:
    mean_rel_l2: float
    median_rel_l2: float
    errors: list[float]


def compute_rel_errors(pred: torch.Tensor, gt: torch.Tensor) -> list[float]:
    pred = pred.detach().cpu()
    gt = gt.detach().cpu()
    errs = []
    for i in range(pred.shape[0]):
        errs.append(rel_l2(pred[i], gt[i]))
    return errs


@torch.no_grad()
def eval_fno(
    model: FNO2d,
    x: torch.Tensor,
    y: torch.Tensor,
    y_normalizer: UnitGaussianNormalizer,
    batch_size: int,
    device: torch.device,
) -> ModelEval:
    model.eval()
    all_errs: list[float] = []
    n = x.shape[0]

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        xb = x[start:end].to(device)
        yb = y[start:end].to(device)

        pred_n = model(xb).squeeze(-1)
        pred = y_normalizer.decode(pred_n)
        all_errs.extend(compute_rel_errors(pred, yb))

    return ModelEval(
        mean_rel_l2=float(np.mean(all_errs)),
        median_rel_l2=float(np.median(all_errs)),
        errors=all_errs,
    )


@torch.no_grad()
def eval_rfno(
    backbone: FNO2d,
    readout: RidgeReadout2D,
    x: torch.Tensor,
    y: torch.Tensor,
    y_normalizer: UnitGaussianNormalizer,
    batch_size: int,
    device: torch.device,
) -> ModelEval:
    backbone.eval()
    all_errs: list[float] = []
    n = x.shape[0]

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        xb = x[start:end].to(device)
        yb = y[start:end].to(device)

        feat = backbone.forward_features(xb)
        pred_n = readout.predict(feat)
        pred = y_normalizer.decode(pred_n)
        all_errs.extend(compute_rel_errors(pred, yb))

    return ModelEval(
        mean_rel_l2=float(np.mean(all_errs)),
        median_rel_l2=float(np.median(all_errs)),
        errors=all_errs,
    )


@torch.no_grad()
def predict_single_fno(
    model: FNO2d,
    x: torch.Tensor,
    y_normalizer: UnitGaussianNormalizer,
    device: torch.device,
) -> torch.Tensor:
    model.eval()
    pred_n = model(x.to(device)).squeeze(-1)
    return y_normalizer.decode(pred_n).cpu()


@torch.no_grad()
def predict_single_rfno(
    backbone: FNO2d,
    readout: RidgeReadout2D,
    x: torch.Tensor,
    y_normalizer: UnitGaussianNormalizer,
    device: torch.device,
) -> torch.Tensor:
    backbone.eval()
    feat = backbone.forward_features(x.to(device))
    pred_n = readout.predict(feat)
    return y_normalizer.decode(pred_n).cpu()


def train_standard_fno(
    args: argparse.Namespace,
    device: torch.device,
    s: int,
    x_train_n: torch.Tensor,
    y_train_n: torch.Tensor,
    y_train: torch.Tensor,
    x_test_n: torch.Tensor,
    y_test: torch.Tensor,
    y_normalizer: UnitGaussianNormalizer,
) -> tuple[FNO2d, list[int], list[float], list[float], ModelEval, ModelEval]:
    model = FNO2d(args.modes, args.modes, args.width).to(device)

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_train_n.reshape(args.ntrain, s, s, 1), y_train_n),
        batch_size=args.batch_size,
        shuffle=True,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    t_max = max(1, args.epochs * len(train_loader))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max)
    criterion = torch.nn.MSELoss()

    epochs_hist: list[int] = []
    train_rel_hist: list[float] = []
    test_rel_hist: list[float] = []

    for ep in range(args.epochs):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            pred = model(xb).squeeze(-1)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            scheduler.step()

        train_eval = eval_fno(model, x_train_n.reshape(args.ntrain, s, s, 1), y_train, y_normalizer, args.batch_size, device)
        test_eval = eval_fno(model, x_test_n.reshape(args.ntest, s, s, 1), y_test, y_normalizer, args.batch_size, device)

        epochs_hist.append(ep)
        train_rel_hist.append(train_eval.mean_rel_l2)
        test_rel_hist.append(test_eval.mean_rel_l2)
        print(
            f"[FNO] epoch={ep:03d} train_relL2={train_eval.mean_rel_l2:.6f} "
            f"test_relL2={test_eval.mean_rel_l2:.6f}"
        )

    final_train = eval_fno(model, x_train_n.reshape(args.ntrain, s, s, 1), y_train, y_normalizer, args.batch_size, device)
    final_test = eval_fno(model, x_test_n.reshape(args.ntest, s, s, 1), y_test, y_normalizer, args.batch_size, device)
    return model, epochs_hist, train_rel_hist, test_rel_hist, final_train, final_test


def fit_reservoir_readout(
    args: argparse.Namespace,
    device: torch.device,
    s: int,
    x_train_n: torch.Tensor,
    y_train_n: torch.Tensor,
) -> tuple[FNO2d, RidgeReadout2D, dict[float, RidgeReadout2D]]:
    backbone = FNO2d(args.modes, args.modes, args.width).to(device)
    backbone.freeze_backbone()
    backbone.eval()

    collector = RidgeReadout2D(in_channels=args.width, ridge_lambda=0.0, stats_device="cpu")

    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_train_n.reshape(args.ntrain, s, s, 1), y_train_n),
        batch_size=args.batch_size,
        shuffle=False,
    )

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            feat = backbone.forward_features(xb)
            collector.update(feat, yb)

    ridge_models: dict[float, RidgeReadout2D] = {}
    for lam in parse_lambdas(args.ridge_lambdas):
        ridge = RidgeReadout2D(in_channels=args.width, ridge_lambda=lam, stats_device="cpu")
        ridge.sum_phi = collector.sum_phi.clone()
        ridge.sum_y = collector.sum_y.clone()
        ridge.sum_phiphi = collector.sum_phiphi.clone()
        ridge.sum_phiy = collector.sum_phiy.clone()
        ridge.count = collector.count
        ridge.solve()
        ridge_models[lam] = ridge

    first_lambda = parse_lambdas(args.ridge_lambdas)[0]
    return backbone, ridge_models[first_lambda], ridge_models


def save_metrics(path: str, metrics: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare Darcy FNO vs Reservoir-FNO")
    parser.add_argument("--config", type=str, default=None, help="JSON config path.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])

    parser.add_argument("--data-source", type=str, default="generate", choices=["mat", "generate"])
    parser.add_argument("--data-mode", type=str, default="separate_files", choices=["single_split", "separate_files"])
    parser.add_argument("--data-file", type=str, default="data/piececonst_r421_N1024_smooth1.mat")
    parser.add_argument("--train-file", type=str, default="data/piececonst_r421_N1024_smooth1.mat")
    parser.add_argument("--test-file", type=str, default="data/piececonst_r421_N1024_smooth2.mat")

    parser.add_argument("--ntrain", type=int, default=1000)
    parser.add_argument("--ntest", type=int, default=100)
    parser.add_argument("--r", type=int, default=5)
    parser.add_argument("--grid-size", type=int, default=421)

    parser.add_argument("--gen-grid-size", type=int, default=32)
    parser.add_argument("--gen-map-type", type=str, default="piecewise", choices=["piecewise", "exp"])
    parser.add_argument("--gen-a-pos", type=float, default=12.0)
    parser.add_argument("--gen-a-neg", type=float, default=3.0)
    parser.add_argument("--gen-alpha", type=float, default=2.0)
    parser.add_argument("--gen-tau", type=float, default=3.0)
    parser.add_argument("--gen-forcing", type=float, default=1.0)
    parser.add_argument("--gen-solver", type=str, default="spsolve", choices=["spsolve", "cg"])
    parser.add_argument("--gen-cg-tol", type=float, default=1e-10)
    parser.add_argument("--gen-cg-maxiter", type=int, default=2000)
    parser.add_argument("--gen-save-mat", type=str, default="")

    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--modes", type=int, default=12)
    parser.add_argument("--width", type=int, default=32)

    parser.add_argument("--ridge-lambdas", type=str, default="0,1e-6,1e-4,1e-2")

    parser.add_argument("--results-root", type=str, default="results")
    parser.add_argument("--run-name", type=str, default="")
    parser.add_argument("--no-viz", action="store_true")

    return parser


def parse_args() -> argparse.Namespace:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=str, default=None)
    pre_args, _ = pre.parse_known_args()

    parser = build_parser()

    if pre_args.config:
        with open(pre_args.config, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        parser.set_defaults(**cfg)

    args = parser.parse_args()
    if args.ntrain <= 0 or args.ntest <= 0:
        parser.error("ntrain/ntest must be positive.")
    return args


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)

    start_time = time.time()
    x_train, y_train, x_test, y_test, s = load_data(args)

    x_normalizer = UnitGaussianNormalizer(x_train)
    y_normalizer = UnitGaussianNormalizer(y_train)

    x_train_n = x_normalizer.encode(x_train)
    x_test_n = x_normalizer.encode(x_test)
    y_train_n = y_normalizer.encode(y_train)

    run_name = args.run_name.strip()
    if not run_name:
        run_name = f"compare_darcy_fno_vs_reservoir_{time.strftime('%Y%m%d_%H%M%S')}"

    out_dir = os.path.join(args.results_root, run_name)
    os.makedirs(out_dir, exist_ok=True)

    print(f"[info] device={device}")
    print(f"[info] data_source={args.data_source} S={s} ntrain={args.ntrain} ntest={args.ntest}")

    fno_model, epochs_hist, train_rel_hist, test_rel_hist, fno_train_eval, fno_test_eval = train_standard_fno(
        args=args,
        device=device,
        s=s,
        x_train_n=x_train_n,
        y_train_n=y_train_n,
        y_train=y_train,
        x_test_n=x_test_n,
        y_test=y_test,
        y_normalizer=y_normalizer,
    )

    backbone, _, ridge_models = fit_reservoir_readout(
        args=args,
        device=device,
        s=s,
        x_train_n=x_train_n,
        y_train_n=y_train_n,
    )

    rfno_results: dict[str, Any] = {}
    best_lambda = None
    best_eval = None

    for lam, ridge in ridge_models.items():
        train_eval = eval_rfno(
            backbone=backbone,
            readout=ridge,
            x=x_train_n.reshape(args.ntrain, s, s, 1),
            y=y_train,
            y_normalizer=y_normalizer,
            batch_size=args.batch_size,
            device=device,
        )
        test_eval = eval_rfno(
            backbone=backbone,
            readout=ridge,
            x=x_test_n.reshape(args.ntest, s, s, 1),
            y=y_test,
            y_normalizer=y_normalizer,
            batch_size=args.batch_size,
            device=device,
        )
        rfno_results[str(lam)] = {
            "train": asdict(train_eval),
            "test": asdict(test_eval),
        }

        print(
            f"[RFNO lambda={lam:g}] train_relL2={train_eval.mean_rel_l2:.6f} "
            f"test_relL2={test_eval.mean_rel_l2:.6f}"
        )

        if best_eval is None or test_eval.mean_rel_l2 < best_eval.mean_rel_l2:
            best_eval = test_eval
            best_lambda = lam

    assert best_eval is not None
    assert best_lambda is not None

    if not args.no_viz:
        plot_learning_curve(
            LearningCurve(
                epochs=epochs_hist,
                train=train_rel_hist,
                test=test_rel_hist,
                train_label="FNO train (relL2)",
                test_label="FNO test (relL2)",
                metric_name="relative L2",
            ),
            out_path_no_ext=os.path.join(out_dir, "fno_learning_curve_relL2"),
            logy=True,
            title="FNO (Darcy 2D)",
        )

        plot_error_histogram(
            fno_test_eval.errors,
            out_path_no_ext=os.path.join(out_dir, "fno_test_relL2_hist"),
            title=f"FNO test relL2 (mean={fno_test_eval.mean_rel_l2:.3g}, median={fno_test_eval.median_rel_l2:.3g})",
        )

        best_rfno_errs = rfno_results[str(best_lambda)]["test"]["errors"]
        plot_error_histogram(
            best_rfno_errs,
            out_path_no_ext=os.path.join(out_dir, "rfno_best_test_relL2_hist"),
            title=(
                f"RFNO test relL2 (lambda={best_lambda:g}, "
                f"mean={best_eval.mean_rel_l2:.3g}, median={best_eval.median_rel_l2:.3g})"
            ),
        )

        sample_id = 0
        x_sample = x_test_n[sample_id : sample_id + 1].reshape(1, s, s, 1)
        y_sample = y_test[sample_id]
        coeff_sample = x_normalizer.decode(x_test_n[sample_id]).cpu()

        fno_pred = predict_single_fno(fno_model, x_sample, y_normalizer, device=device)[0]
        plot_2d_comparison(
            gt=y_sample,
            pred=fno_pred,
            input_field=coeff_sample,
            out_path_no_ext=os.path.join(out_dir, "sample0_fno"),
            suptitle=f"FNO sample0 relL2={rel_l2(fno_pred, y_sample):.3g}",
        )

        best_ridge = ridge_models[best_lambda]
        rfno_pred = predict_single_rfno(backbone, best_ridge, x_sample, y_normalizer, device=device)[0]
        plot_2d_comparison(
            gt=y_sample,
            pred=rfno_pred,
            input_field=coeff_sample,
            out_path_no_ext=os.path.join(out_dir, "sample0_rfno_best"),
            suptitle=(
                f"RFNO sample0 lambda={best_lambda:g} "
                f"relL2={rel_l2(rfno_pred, y_sample):.3g}"
            ),
        )

    elapsed = time.time() - start_time

    metrics = {
        "task": "darcy_2d_fno_vs_reservoir",
        "elapsed_sec": elapsed,
        "device": str(device),
        "data_source": args.data_source,
        "grid_size_after_downsample": s,
        "ntrain": args.ntrain,
        "ntest": args.ntest,
        "fno": {
            "train": asdict(fno_train_eval),
            "test": asdict(fno_test_eval),
        },
        "rfno": {
            "best_lambda": best_lambda,
            "best_test": asdict(best_eval),
            "all": rfno_results,
        },
        "config": vars(args),
    }

    metrics_path = os.path.join(out_dir, "metrics.json")
    save_metrics(metrics_path, metrics)

    print("=" * 72)
    print(
        "FNO  test relL2: "
        f"mean={fno_test_eval.mean_rel_l2:.6f} median={fno_test_eval.median_rel_l2:.6f}"
    )
    print(
        "RFNO test relL2: "
        f"mean={best_eval.mean_rel_l2:.6f} median={best_eval.median_rel_l2:.6f} "
        f"(best lambda={best_lambda:g})"
    )
    print(f"metrics: {metrics_path}")


if __name__ == "__main__":
    main()
