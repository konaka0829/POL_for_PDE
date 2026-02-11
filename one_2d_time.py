import argparse
import os

import numpy as np
import torch
from timeit import default_timer

from cli_utils import add_data_mode_args, add_split_args, validate_data_mode_args
from one.one_config import add_one_optical_args
from one.one_models import ONE2dTimeNS
from viz_utils import (
    LearningCurve,
    plot_2d_time_slices,
    plot_error_histogram,
    plot_learning_curve,
    plot_rel_l2_over_time,
    rel_l2,
)


def _count_params(model: torch.nn.Module) -> int:
    total = 0
    for p in model.parameters():
        total += p.numel() * (2 if p.is_complex() else 1)
    return total


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ONE baseline for Navier-Stokes (2D time)")
    add_data_mode_args(
        parser,
        default_data_mode="separate_files",
        default_data_file="data/ns_data_V100_N1000_T50_1.mat",
        default_train_file="data/ns_data_V100_N1000_T50_1.mat",
        default_test_file="data/ns_data_V100_N1000_T50_2.mat",
    )
    parser.add_argument("--ntrain", type=int, default=1000)
    parser.add_argument("--ntest", type=int, default=200)
    parser.add_argument("--width", type=int, default=20)
    parser.add_argument("--activation", choices=("tanh", "gelu"), default="tanh")
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--sub", type=int, default=1)
    parser.add_argument("--S", type=int, default=64)
    parser.add_argument("--T-in", type=int, default=10)
    parser.add_argument("--T", type=int, default=40)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--smoke-test", action="store_true", help="Run one synthetic forward/backward pass and exit.")
    add_split_args(parser, default_train_split=0.8, default_seed=0)
    add_one_optical_args(parser)
    parser.set_defaults(domain_padding=0)
    return parser


def _validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    validate_data_mode_args(args, parser)


def _run_smoke_test(args: argparse.Namespace, device: torch.device) -> None:
    s = 32
    t_in = 10
    t = 4
    step = 1

    domain_padding = args.domain_padding
    if args.one_mode == "tp_compat" and domain_padding == 0:
        domain_padding = 2

    model = ONE2dTimeNS(
        spatial_size=s,
        input_steps=t_in,
        width=8,
        domain_padding=domain_padding,
        activation=args.activation,
        mode=args.one_mode,
        donn_ratio=args.donn_ratio,
        wavelength=args.wavelength,
        pixel_size=args.pixel_size,
        distance=args.distance,
        phase_init=args.phase_init,
        xbar_noise_std=args.xbar_noise_std,
        prop_padding=args.donn_prop_padding,
        donn_projection=args.donn_projection,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    xx = torch.randn(2, s, s, t_in, device=device)
    yy = torch.randn(2, s, s, t, device=device)
    bs = xx.shape[0]

    loss = 0.0
    for ti in range(0, t, step):
        y = yy[..., ti : ti + step]
        im = model(xx)
        loss = loss + torch.mean((im.reshape(bs, -1) - y.reshape(bs, -1)) ** 2)
        xx = torch.cat((xx[..., step:], im), dim=-1)

    opt.zero_grad()
    loss.backward()
    opt.step()
    print(f"[smoke-test] success: loss={float(loss):.6f}")


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    _validate_args(args, parser)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.smoke_test:
        _run_smoke_test(args, device)
        return

    from utilities3 import LpLoss, MatReader

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    ntrain = args.ntrain
    ntest = args.ntest

    batch_size = args.batch_size
    learning_rate = args.learning_rate
    epochs = args.epochs
    iterations = epochs * (ntrain // batch_size)

    sub = args.sub
    S = args.S
    T_in = args.T_in
    T = args.T
    step = args.step
    domain_padding = args.domain_padding
    if args.one_mode == "tp_compat" and domain_padding == 0:
        domain_padding = 2

    if args.data_mode == "single_split":
        reader = MatReader(args.data_file)
        full_u = reader.read_field("u")
        total = full_u.shape[0]
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

        train_a = full_u[train_idx, ::sub, ::sub, :T_in]
        train_u = full_u[train_idx, ::sub, ::sub, T_in : T + T_in]
        test_a = full_u[test_idx, ::sub, ::sub, :T_in]
        test_u = full_u[test_idx, ::sub, ::sub, T_in : T + T_in]
    else:
        reader = MatReader(args.train_file)
        train_a = reader.read_field("u")[:ntrain, ::sub, ::sub, :T_in]
        train_u = reader.read_field("u")[:ntrain, ::sub, ::sub, T_in : T + T_in]

        reader = MatReader(args.test_file)
        test_a = reader.read_field("u")[-ntest:, ::sub, ::sub, :T_in]
        test_u = reader.read_field("u")[-ntest:, ::sub, ::sub, T_in : T + T_in]

    print(train_u.shape)
    print(test_u.shape)
    assert S == train_u.shape[-2]
    assert T == train_u.shape[-1]

    train_a = train_a.reshape(ntrain, S, S, T_in)
    test_a = test_a.reshape(ntest, S, S, T_in)

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(train_a, train_u), batch_size=batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(test_a, test_u), batch_size=batch_size, shuffle=False
    )

    model = ONE2dTimeNS(
        spatial_size=S,
        input_steps=T_in,
        width=args.width,
        domain_padding=domain_padding,
        activation=args.activation,
        mode=args.one_mode,
        donn_ratio=args.donn_ratio,
        wavelength=args.wavelength,
        pixel_size=args.pixel_size,
        distance=args.distance,
        phase_init=args.phase_init,
        xbar_noise_std=args.xbar_noise_std,
        prop_padding=args.donn_prop_padding,
        donn_projection=args.donn_projection,
    ).to(device)
    print(_count_params(model))

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iterations)

    myloss = LpLoss(size_average=False)
    viz_dir = os.path.join("visualizations", f"one_2d_time_{args.one_mode}")
    os.makedirs(viz_dir, exist_ok=True)
    hist_epochs = []
    hist_train_step = []
    hist_train_full = []
    hist_test_step = []
    hist_test_full = []

    for ep in range(epochs):
        model.train()
        t1 = default_timer()
        train_l2_step = 0.0
        train_l2_full = 0.0

        for xx, yy in train_loader:
            xx = xx.to(device)
            yy = yy.to(device)
            bs = xx.shape[0]

            loss = 0.0
            for ti in range(0, T, step):
                y = yy[..., ti : ti + step]
                im = model(xx)
                loss = loss + myloss(im.reshape(bs, -1), y.reshape(bs, -1))

                if ti == 0:
                    pred = im
                else:
                    pred = torch.cat((pred, im), dim=-1)

                xx = torch.cat((xx[..., step:], im), dim=-1)

            train_l2_step += loss.item()
            train_l2_full += myloss(pred.reshape(bs, -1), yy.reshape(bs, -1)).item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        model.eval()
        test_l2_step = 0.0
        test_l2_full = 0.0
        with torch.no_grad():
            for xx, yy in test_loader:
                xx = xx.to(device)
                yy = yy.to(device)
                bs = xx.shape[0]

                loss = 0.0
                for ti in range(0, T, step):
                    y = yy[..., ti : ti + step]
                    im = model(xx)
                    loss = loss + myloss(im.reshape(bs, -1), y.reshape(bs, -1))

                    if ti == 0:
                        pred = im
                    else:
                        pred = torch.cat((pred, im), dim=-1)

                    xx = torch.cat((xx[..., step:], im), dim=-1)

                test_l2_step += loss.item()
                test_l2_full += myloss(pred.reshape(bs, -1), yy.reshape(bs, -1)).item()

        t2 = default_timer()
        hist_epochs.append(ep)
        hist_train_step.append(train_l2_step / ntrain / (T / step))
        hist_train_full.append(train_l2_full / ntrain)
        hist_test_step.append(test_l2_step / ntest / (T / step))
        hist_test_full.append(test_l2_full / ntest)
        print(
            ep,
            t2 - t1,
            train_l2_step / ntrain / (T / step),
            train_l2_full / ntrain,
            test_l2_step / ntest / (T / step),
            test_l2_full / ntest,
        )

    try:
        plot_learning_curve(
            LearningCurve(
                epochs=hist_epochs,
                train=hist_train_step,
                test=hist_test_step,
                train_label="train (step relL2)",
                test_label="test (step relL2)",
                metric_name="relative L2",
            ),
            out_path_no_ext=os.path.join(viz_dir, "learning_curve_step_relL2"),
            logy=True,
            title=f"one_2d_time ({args.one_mode}): stepwise relative L2",
        )
        plot_learning_curve(
            LearningCurve(
                epochs=hist_epochs,
                train=hist_train_full,
                test=hist_test_full,
                train_label="train (full relL2)",
                test_label="test (full relL2)",
                metric_name="relative L2",
            ),
            out_path_no_ext=os.path.join(viz_dir, "learning_curve_full_relL2"),
            logy=True,
            title=f"one_2d_time ({args.one_mode}): full-trajectory relative L2",
        )

        def _rollout_autoregressive(xx0: torch.Tensor) -> torch.Tensor:
            xx = xx0.to(device)
            preds = []
            for _ in range(0, T, step):
                im = model(xx)
                preds.append(im)
                xx = torch.cat((xx[..., step:], im), dim=-1)
            return torch.cat(preds, dim=-1)

        model.eval()
        sample_ids = [0, min(1, ntest - 1), min(2, ntest - 1)]
        t_indices = [0, T // 2, T - 1]
        with torch.no_grad():
            for i in sample_ids:
                gt_i = test_u[i].cpu()
                pred_i = _rollout_autoregressive(test_a[i : i + 1]).squeeze(0).cpu()
                e_full = rel_l2(pred_i, gt_i)
                plot_2d_time_slices(
                    gt=gt_i,
                    pred=pred_i,
                    t_indices=t_indices,
                    out_path_no_ext=os.path.join(viz_dir, f"sample_{i:03d}_slices"),
                    suptitle=f"sample {i}  full relL2={e_full:.3g}",
                )
                plot_rel_l2_over_time(
                    gt=gt_i,
                    pred=pred_i,
                    out_path_no_ext=os.path.join(viz_dir, f"sample_{i:03d}_relL2_over_time"),
                )

        n_hist = min(ntest, 50)
        per_sample_full = []
        with torch.no_grad():
            for i in range(n_hist):
                pred_i = _rollout_autoregressive(test_a[i : i + 1]).squeeze(0).cpu()
                per_sample_full.append(rel_l2(pred_i, test_u[i].cpu()))
        plot_error_histogram(
            per_sample_full,
            os.path.join(viz_dir, f"test_full_relL2_hist_first{n_hist}"),
            title=f"full relL2 histogram (first {n_hist} test samples)",
        )
    except Exception as e:
        print(f"[viz] failed: {e}")


if __name__ == "__main__":
    main()
