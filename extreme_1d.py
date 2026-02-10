import argparse
import os
import time

import numpy as np
import torch

from cli_utils import add_data_mode_args, add_split_args, validate_data_mode_args
from extremonet.io import save_eon
from extremonet.model import ExtremeLearning, ExtremONet
from extremonet.train import train_ridge
from operator_data import eon_pkl_to_points, load_pickle, load_mat_dict, mat_to_points
from viz_utils import LearningCurve, plot_1d_prediction, plot_error_histogram, plot_learning_curve, rel_l2


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ExtremONet 1D")
    add_data_mode_args(
        parser,
        default_data_mode="single_split",
        default_data_file="data/burgers_data_R10.mat",
        default_train_file=None,
        default_test_file=None,
    )
    add_split_args(parser, default_train_split=0.8, default_seed=0)

    parser.add_argument("--dataset-format", choices=("mat", "eon_pkl"), default="mat")
    parser.add_argument("--eon-data-file", default=None)
    parser.add_argument("--eon-train-file", default=None)
    parser.add_argument("--eon-test-file", default=None)
    parser.add_argument("--eon-meta-file", default=None)

    parser.add_argument("--ntrain", type=int, default=1000)
    parser.add_argument("--ntest", type=int, default=100)
    parser.add_argument("--sub", type=int, default=2**3)

    parser.add_argument("--num-sensors", type=int, default=128)
    parser.add_argument("--sensor-strategy", choices=("uniform_idx", "random_idx"), default="uniform_idx")
    parser.add_argument("--num-query", type=int, default=256)
    parser.add_argument("--query-strategy", choices=("all_grid", "random"), default="random")

    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--trunk-c", type=int, default=1)
    parser.add_argument("--branch-c", type=int, default=1)
    parser.add_argument("--trunk-s", type=float, default=10.0)
    parser.add_argument("--branch-s", type=float, default=0.1)

    parser.add_argument("--ridge-bounds", nargs=2, type=float, default=[-10.0, 10.0])
    parser.add_argument("--ridge-iters", type=int, default=200)
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=64)

    parser.add_argument("--run-name", default="")
    return parser


def _slice_dict(d: dict[str, torch.Tensor], idx: np.ndarray) -> dict[str, torch.Tensor]:
    return {k: v[idx] for k, v in d.items()}


def _train_test_indices(total: int, ntrain: int, ntest: int, train_split: float, shuffle: bool, seed: int):
    idx = np.arange(total)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(idx)
    split = int(total * train_split)
    tr = idx[:split]
    te = idx[split:]
    if ntrain > len(tr) or ntest > len(te):
        raise ValueError(f"Not enough samples: total={total}, train={len(tr)}, test={len(te)}")
    return tr[:ntrain], te[:ntest]


def _load_points(args):
    if args.dataset_format == "mat":
        if args.data_mode == "single_split":
            raw = load_mat_dict(args.data_file, ["a", "u"])
            tr_idx, te_idx = _train_test_indices(
                raw["a"].shape[0], args.ntrain, args.ntest, args.train_split, args.shuffle, args.seed
            )
            tr = _slice_dict(raw, tr_idx)
            te = _slice_dict(raw, te_idx)
        else:
            tr = {k: v[: args.ntrain] for k, v in load_mat_dict(args.train_file, ["a", "u"]).items()}
            te = {k: v[: args.ntest] for k, v in load_mat_dict(args.test_file, ["a", "u"]).items()}

        xq_tr, us_tr, yq_tr, meta_tr = mat_to_points(
            tr,
            "burgers",
            num_sensors=args.num_sensors,
            sensor_strategy=args.sensor_strategy,
            num_query=args.num_query,
            query_strategy=args.query_strategy,
            seed=args.seed,
            sub=args.sub,
        )
        xq_te, us_te, yq_te, meta_te = mat_to_points(
            te,
            "burgers",
            num_sensors=args.num_sensors,
            sensor_strategy=args.sensor_strategy,
            num_query=args.num_query,
            query_strategy=args.query_strategy,
            seed=args.seed + 1,
            sub=args.sub,
            sensor_indices=meta_tr["sensor_indices"],
            query_indices=meta_tr["query_indices"],
        )
        return xq_tr, us_tr, yq_tr, xq_te, us_te, yq_te

    if args.data_mode == "single_split":
        eon_file = args.eon_data_file or args.data_file
        t, y, u = eon_pkl_to_points(load_pickle(eon_file))
        tr_idx, te_idx = _train_test_indices(t.shape[0], args.ntrain, args.ntest, args.train_split, args.shuffle, args.seed)
        t_tr, y_tr, u_tr = t[tr_idx], y[tr_idx], u[tr_idx]
        t_te, y_te, u_te = t[te_idx], y[te_idx], u[te_idx]
    else:
        tr_file = args.eon_train_file or args.train_file
        te_file = args.eon_test_file or args.test_file
        t_tr, y_tr, u_tr = eon_pkl_to_points(load_pickle(tr_file))
        t_te, y_te, u_te = eon_pkl_to_points(load_pickle(te_file))
        t_tr, y_tr, u_tr = t_tr[: args.ntrain], y_tr[: args.ntrain], u_tr[: args.ntrain]
        t_te, y_te, u_te = t_te[: args.ntest], y_te[: args.ntest], u_te[: args.ntest]

    # External EON datasets store one (t, u) pair per row and y as vector output.
    # Convert to (B, Q, D) with Q=1 to match the batched query interface used here.
    if y_tr.ndim == 1:
        y_tr = y_tr.view(-1, 1, 1)
    elif y_tr.ndim == 2:
        y_tr = y_tr.unsqueeze(1)
    if y_te.ndim == 1:
        y_te = y_te.view(-1, 1, 1)
    elif y_te.ndim == 2:
        y_te = y_te.unsqueeze(1)

    xq_tr = t_tr.unsqueeze(1) if t_tr.ndim == 2 else t_tr
    xq_te = t_te.unsqueeze(1) if t_te.ndim == 2 else t_te
    return xq_tr, u_tr, y_tr, xq_te, u_te, y_te


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    validate_data_mode_args(args, parser)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    xq_tr, us_tr, yq_tr, xq_te, us_te, yq_te = _load_points(args)

    trunk_indim = xq_tr.shape[-1]
    branch_indim = us_tr.shape[-1]
    outdim = yq_tr.shape[-1]

    trunk_mean = xq_tr.reshape(-1, trunk_indim).mean(dim=0)
    trunk_std = xq_tr.reshape(-1, trunk_indim).std(dim=0) + 1e-6
    branch_mean = us_tr.mean(dim=0)
    branch_std = us_tr.std(dim=0) + 1e-6

    model = ExtremONet(
        trunk=ExtremeLearning(
            trunk_indim,
            args.width,
            c=args.trunk_c,
            s=args.trunk_s,
            norm_mean=trunk_mean,
            norm_std=trunk_std,
            seed=args.seed,
        ),
        branch=ExtremeLearning(
            branch_indim,
            args.width,
            c=args.branch_c,
            s=args.branch_s,
            norm_mean=branch_mean,
            norm_std=branch_std,
            seed=args.seed + 1,
        ),
        outdim=outdim,
    ).to(device)

    result = train_ridge(
        model,
        xq_tr,
        us_tr,
        yq_tr,
        bounds=(args.ridge_bounds[0], args.ridge_bounds[1]),
        iters=args.ridge_iters,
        val_frac=args.val_frac,
        seed=args.seed,
        batch_size=args.batch_size,
    )

    model.eval()
    with torch.no_grad():
        pred_te = model.predict_tensor(xq_te.to(device), us_te.to(device)).cpu()

    sample_errors = [rel_l2(pred_te[i], yq_te[i]) for i in range(pred_te.shape[0])]
    print(f"test relL2 mean={float(np.mean(sample_errors)):.6e}, median={float(np.median(sample_errors)):.6e}")

    run_name = args.run_name or f"extreme_1d_{args.dataset_format}_{time.strftime('%Y%m%d_%H%M%S')}"
    os.makedirs("model", exist_ok=True)
    os.makedirs("pred", exist_ok=True)
    os.makedirs("viz", exist_ok=True)
    viz_dir = os.path.join("viz", run_name)
    os.makedirs(viz_dir, exist_ok=True)

    ckpt_path = os.path.join("model", f"{run_name}.pt")
    save_eon(
        model,
        ckpt_path,
        extra_meta={
            "dataset_format": args.dataset_format,
            "num_sensors": args.num_sensors,
            "num_query": args.num_query,
        },
    )
    torch.save(
        {
            "pred": pred_te,
            "gt": yq_te,
            "x_query": xq_te,
            "u_sensors": us_te,
            "errors": sample_errors,
        },
        os.path.join("pred", f"{run_name}.pt"),
    )

    plot_learning_curve(
        LearningCurve(
            epochs=list(range(len(result.history_val_nmse))),
            train=result.history_train_nmse,
            test=result.history_val_nmse,
            train_label="train (NMSE)",
            test_label="val (NMSE)",
            metric_name="NMSE",
        ),
        out_path_no_ext=os.path.join(viz_dir, "ridge_search_nmse"),
        logy=True,
        title="extreme_1d ridge search",
    )
    plot_error_histogram(sample_errors, os.path.join(viz_dir, "test_relL2_hist"))

    if yq_te.shape[-1] == 1:
        if xq_te.ndim == 2:
            x_axis = xq_te[:, 0].cpu().numpy()
        elif xq_te.ndim == 3:
            x_axis = xq_te[0, :, 0].cpu().numpy()
        else:
            raise ValueError(f"Unexpected xq_te ndim: {xq_te.ndim}")
        order = np.argsort(x_axis)
        for i in [0, min(1, len(sample_errors) - 1), min(2, len(sample_errors) - 1)]:
            plot_1d_prediction(
                x=x_axis[order],
                gt=yq_te[i, :, 0].cpu().numpy()[order],
                pred=pred_te[i, :, 0].cpu().numpy()[order],
                input_u0=None,
                out_path_no_ext=os.path.join(viz_dir, f"sample_{i:03d}"),
                title_prefix=f"sample {i}: ",
            )


if __name__ == "__main__":
    main()
