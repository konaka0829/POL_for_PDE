import argparse
import os
from timeit import default_timer

import numpy as np
import torch

from cli_utils import add_data_mode_args, add_split_args, validate_data_mode_args
from fno_generic import FNO2dGeneric
from semigroup_layers import HeatSemigroup2d
from utilities3 import LpLoss, MatReader, device
from viz_utils import (
    LearningCurve,
    plot_2d_time_slices,
    plot_error_histogram,
    plot_learning_curve,
    plot_rel_l2_over_time,
    rel_l2,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Conjugacy Learning with Heat Semigroup (2D time)")
    add_data_mode_args(
        parser,
        default_data_mode="separate_files",
        default_data_file="data/ns_data_V100_N1000_T50_1.mat",
        default_train_file="data/ns_data_V100_N1000_T50_1.mat",
        default_test_file="data/ns_data_V100_N1000_T50_2.mat",
    )
    add_split_args(parser, default_train_split=0.8, default_seed=0)

    parser.add_argument("--ntrain", type=int, default=1000, help="Number of training samples.")
    parser.add_argument("--ntest", type=int, default=200, help="Number of test samples.")
    parser.add_argument("--batch-size", type=int, default=20, help="Batch size.")
    parser.add_argument("--epochs", type=int, default=500, help="Number of epochs.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate.")

    parser.add_argument("--sub", type=int, default=1, help="Spatial subsampling rate.")
    parser.add_argument("--S", type=int, default=64, help="Expected post-subsampling grid size.")
    parser.add_argument("--sub-t", type=int, default=1, help="Temporal subsampling rate.")
    parser.add_argument("--T", type=int, default=40, help="Prediction horizon (window length is T+1).")
    parser.add_argument("--dt", type=float, default=1.0, help="Time step per latent semigroup application.")

    parser.add_argument("--modes", type=int, default=12, help="Number of Fourier modes.")
    parser.add_argument("--width", type=int, default=20, help="FNO width.")
    parser.add_argument("--cz", type=int, default=8, help="Latent channel dimension.")

    parser.add_argument("--mu", type=float, default=0.1, help="Weight for semigroup consistency loss.")
    parser.add_argument("--mu-warmup-epochs", type=int, default=50, help="Linear warmup epochs for mu. 0 disables.")
    parser.add_argument("--lambda-ae", type=float, default=1.0, help="Weight for auto-encoding loss.")

    parser.add_argument("--nu", type=float, default=0.01, help="Heat semigroup diffusion coefficient.")
    parser.add_argument("--learn-nu", action="store_true", help="Make nu learnable (positive via softplus).")
    parser.add_argument(
        "--use-2pi",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use 2*pi scaling for FFT frequencies.",
    )
    parser.add_argument("--domain-length", type=float, default=1.0, help="Physical domain length.")

    parser.add_argument(
        "--use-instance-norm",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable InstanceNorm in 2D FNO blocks.",
    )
    parser.add_argument(
        "--random-t0",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Sample random start time per batch.",
    )

    return parser


def _validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    validate_data_mode_args(args, parser)
    if args.ntrain <= 0 or args.ntest <= 0:
        parser.error("--ntrain and --ntest must be positive")
    if args.batch_size <= 0 or args.epochs <= 0:
        parser.error("--batch-size and --epochs must be positive")
    if args.sub <= 0 or args.sub_t <= 0:
        parser.error("--sub and --sub-t must be positive")
    if args.S <= 0:
        parser.error("--S must be positive")
    if args.T < 1:
        parser.error("--T must be >= 1")
    if args.dt <= 0:
        parser.error("--dt must be positive")
    if args.cz <= 0:
        parser.error("--cz must be positive")
    if args.mu_warmup_epochs < 0:
        parser.error("--mu-warmup-epochs must be >= 0")


def _load_u_full(path: str, sub: int, sub_t: int) -> torch.Tensor:
    reader = MatReader(path)
    u_full = reader.read_field("u").float()
    if u_full.ndim != 4:
        raise ValueError(f"expected u with shape (N,S,S,T), got {tuple(u_full.shape)}")
    return u_full[:, ::sub, ::sub, ::sub_t]


def _split_single_tensor(
    full: torch.Tensor,
    ntrain: int,
    ntest: int,
    train_split: float,
    seed: int,
    shuffle: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    total = full.shape[0]
    idx = np.arange(total)
    rng = np.random.default_rng(seed)
    if shuffle:
        rng.shuffle(idx)

    split_idx = int(total * train_split)
    train_idx = idx[:split_idx]
    test_idx = idx[split_idx:]

    if ntrain > len(train_idx) or ntest > len(test_idx):
        raise ValueError(
            f"Not enough samples for ntrain={ntrain}, ntest={ntest} with train split {train_split} "
            f"(total={total}, train={len(train_idx)}, test={len(test_idx)})"
        )

    return full[train_idx[:ntrain]], full[test_idx[:ntest]]


def _sample_window(u_full: torch.Tensor, T: int, random_t0: bool) -> torch.Tensor:
    # u_full: (B, S, S, T_total)
    t_total = u_full.shape[-1]
    if t_total < T + 1:
        raise ValueError(f"Need at least T+1={T+1} timesteps, got T_total={t_total}")
    max_t0 = t_total - (T + 1)
    if random_t0 and max_t0 > 0:
        t0 = torch.randint(low=0, high=max_t0 + 1, size=(1,), device=u_full.device).item()
    else:
        t0 = 0
    return u_full[..., t0 : t0 + T + 1]


def _rel_lp_loss(myloss: LpLoss, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    b = pred.shape[0]
    return myloss(pred.reshape(b, -1), gt.reshape(b, -1))


def _sg_loss_2d(z_enc: torch.Tensor, z_ref: torch.Tensor) -> torch.Tensor:
    # Sum over batch of per-sample mean squared error in latent space.
    return ((z_enc - z_ref) ** 2).mean(dim=(1, 2, 3)).sum()


def _rollout_2d(
    enc: torch.nn.Module,
    dec: torch.nn.Module,
    heat: torch.nn.Module,
    u0: torch.Tensor,
    T: int,
    dt: float,
) -> torch.Tensor:
    # u0: (B,S,S,1) -> output (B,S,S,T+1)
    z = enc(u0)
    preds = []
    for n in range(T + 1):
        if n > 0:
            z = heat(z, dt)
        preds.append(dec(z))
    return torch.cat(preds, dim=-1)


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    _validate_args(args, parser)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.data_mode == "single_split":
        full = _load_u_full(args.data_file, sub=args.sub, sub_t=args.sub_t)
        train_u, test_u = _split_single_tensor(
            full,
            ntrain=args.ntrain,
            ntest=args.ntest,
            train_split=args.train_split,
            seed=args.seed,
            shuffle=args.shuffle,
        )
    else:
        train_all = _load_u_full(args.train_file, sub=args.sub, sub_t=args.sub_t)
        test_all = _load_u_full(args.test_file, sub=args.sub, sub_t=args.sub_t)
        if args.ntrain > train_all.shape[0] or args.ntest > test_all.shape[0]:
            raise ValueError(
                f"Requested ntrain={args.ntrain}, ntest={args.ntest}, "
                f"but train/test sizes are {train_all.shape[0]} and {test_all.shape[0]}"
            )
        train_u = train_all[: args.ntrain]
        test_u = test_all[-args.ntest :]

    if train_u.shape[1] != args.S or train_u.shape[2] != args.S:
        raise ValueError(
            f"After subsampling expected S={args.S}, but got train shape {tuple(train_u.shape)}"
        )
    if test_u.shape[1] != args.S or test_u.shape[2] != args.S:
        raise ValueError(
            f"After subsampling expected S={args.S}, but got test shape {tuple(test_u.shape)}"
        )

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(train_u), batch_size=args.batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(test_u), batch_size=args.batch_size, shuffle=False
    )

    enc = FNO2dGeneric(
        in_dim=1,
        out_dim=args.cz,
        modes1=args.modes,
        modes2=args.modes,
        width=args.width,
        use_grid=True,
        use_instance_norm=args.use_instance_norm,
    ).to(device)
    dec = FNO2dGeneric(
        in_dim=args.cz,
        out_dim=1,
        modes1=args.modes,
        modes2=args.modes,
        width=args.width,
        use_grid=True,
        use_instance_norm=args.use_instance_norm,
    ).to(device)
    heat = HeatSemigroup2d(
        nu=args.nu,
        learnable_nu=args.learn_nu,
        domain_length=args.domain_length,
        use_2pi=args.use_2pi,
    ).to(device)

    params = list(enc.parameters()) + list(dec.parameters()) + list(heat.parameters())
    optimizer = torch.optim.Adam(params, lr=args.learning_rate, weight_decay=1e-4)
    iterations = max(1, args.epochs * max(1, args.ntrain // args.batch_size))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iterations)

    myloss = LpLoss(size_average=False)

    run_name = (
        f"conjugacy_2d_time_n{args.ntrain}_T{args.T}_m{args.modes}_w{args.width}"
        f"_cz{args.cz}_mu{args.mu}_muw{args.mu_warmup_epochs}_lae{args.lambda_ae}"
        f"_nu{args.nu}{'_lnu' if args.learn_nu else ''}"
    )
    image_dir = os.path.join("image", run_name)
    os.makedirs(image_dir, exist_ok=True)

    hist_ep = []
    hist_train_pred = []
    hist_train_total = []
    hist_test_pred = []
    hist_test_total = []

    for ep in range(args.epochs):
        t1 = default_timer()
        if args.mu_warmup_epochs <= 0:
            mu_eff = args.mu
        else:
            mu_eff = args.mu * min(1.0, ep / args.mu_warmup_epochs)
        enc.train()
        dec.train()
        heat.train()

        train_pred_sum = 0.0
        train_total_sum = 0.0
        train_count = 0

        for (u_batch,) in train_loader:
            u_batch = u_batch.to(device)
            u_gt = _sample_window(u_batch, T=args.T, random_t0=args.random_t0)

            u0 = u_gt[..., 0].unsqueeze(-1)
            z = enc(u0)

            pred_acc = 0.0
            sg_acc = 0.0
            ae_acc = 0.0
            ae_acc = ae_acc + _rel_lp_loss(myloss, dec(z), u0)  # n=0 AE only

            for n in range(1, args.T + 1):
                z = heat(z, args.dt)
                gt_n = u_gt[..., n].unsqueeze(-1)
                u_hat = dec(z)
                z_enc = enc(gt_n)

                pred_acc = pred_acc + _rel_lp_loss(myloss, u_hat, gt_n)
                sg_acc = sg_acc + _sg_loss_2d(z_enc, z)
                ae_acc = ae_acc + _rel_lp_loss(myloss, dec(z_enc), gt_n)

            pred_term = pred_acc / args.T
            sg_term = sg_acc / args.T
            ae_term = ae_acc / (args.T + 1)
            loss = pred_term + mu_eff * sg_term + args.lambda_ae * ae_term

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            b = u_gt.shape[0]
            train_pred_sum += pred_acc.item()
            train_total_sum += loss.item()
            train_count += b

        enc.eval()
        dec.eval()
        heat.eval()

        test_pred_sum = 0.0
        test_total_sum = 0.0
        test_count = 0

        with torch.no_grad():
            for (u_batch,) in test_loader:
                u_batch = u_batch.to(device)
                u_gt = _sample_window(u_batch, T=args.T, random_t0=False)

                u0 = u_gt[..., 0].unsqueeze(-1)
                z = enc(u0)

                pred_acc = 0.0
                sg_acc = 0.0
                ae_acc = 0.0
                ae_acc = ae_acc + _rel_lp_loss(myloss, dec(z), u0)  # n=0 AE only

                for n in range(1, args.T + 1):
                    z = heat(z, args.dt)
                    gt_n = u_gt[..., n].unsqueeze(-1)
                    u_hat = dec(z)
                    z_enc = enc(gt_n)

                    pred_acc = pred_acc + _rel_lp_loss(myloss, u_hat, gt_n)
                    sg_acc = sg_acc + _sg_loss_2d(z_enc, z)
                    ae_acc = ae_acc + _rel_lp_loss(myloss, dec(z_enc), gt_n)

                pred_term = pred_acc / args.T
                sg_term = sg_acc / args.T
                ae_term = ae_acc / (args.T + 1)
                loss = pred_term + mu_eff * sg_term + args.lambda_ae * ae_term

                b = u_gt.shape[0]
                test_pred_sum += pred_acc.item()
                test_total_sum += loss.item()
                test_count += b

        train_pred_epoch = train_pred_sum / max(1, train_count * args.T)
        train_total_epoch = train_total_sum / max(1, train_count)
        test_pred_epoch = test_pred_sum / max(1, test_count * args.T)
        test_total_epoch = test_total_sum / max(1, test_count)

        hist_ep.append(ep)
        hist_train_pred.append(train_pred_epoch)
        hist_train_total.append(train_total_epoch)
        hist_test_pred.append(test_pred_epoch)
        hist_test_total.append(test_total_epoch)

        t2 = default_timer()
        nu_val = heat.nu_value(device=device, dtype=torch.float32).item()
        print(
            ep,
            f"time={t2 - t1:.2f}s",
            f"train_pred={train_pred_epoch:.4e}",
            f"train_total={train_total_epoch:.4e}",
            f"test_pred={test_pred_epoch:.4e}",
            f"test_total={test_total_epoch:.4e}",
            f"mu_eff={mu_eff:.4e}",
            f"nu={nu_val:.4e}",
        )

    try:
        plot_learning_curve(
            LearningCurve(
                epochs=hist_ep,
                train=hist_train_pred,
                test=hist_test_pred,
                train_label="train (pred relL2)",
                test_label="test (pred relL2)",
                metric_name="relative L2",
            ),
            out_path_no_ext=os.path.join(image_dir, "learning_curve_pred_relL2"),
            logy=True,
            title="conjugacy_2d_time: prediction relative L2",
        )

        plot_learning_curve(
            LearningCurve(
                epochs=hist_ep,
                train=hist_train_total,
                test=hist_test_total,
                train_label="train (total)",
                test_label="test (total)",
                metric_name="total loss",
            ),
            out_path_no_ext=os.path.join(image_dir, "learning_curve_total"),
            logy=True,
            title="conjugacy_2d_time: total loss",
        )
    except Exception as exc:
        print(f"[viz] failed to draw learning curves: {exc}")

    try:
        enc.eval()
        dec.eval()
        heat.eval()

        n_hist = min(args.ntest, 50)
        per_sample = []
        with torch.no_grad():
            for i in range(n_hist):
                sample = test_u[i : i + 1].to(device)
                gt_seq = _sample_window(sample, T=args.T, random_t0=False)
                pred_seq = _rollout_2d(enc, dec, heat, gt_seq[..., 0].unsqueeze(-1), args.T, args.dt)
                per_sample.append(rel_l2(pred_seq.squeeze(0).cpu(), gt_seq.squeeze(0).cpu()))

        plot_error_histogram(
            per_sample,
            out_path_no_ext=os.path.join(image_dir, f"test_full_relL2_hist_first{n_hist}"),
            title=f"full trajectory relL2 (first {n_hist} test samples)",
        )

        sample_ids = [0, min(1, args.ntest - 1), min(2, args.ntest - 1)]
        t_indices = [0, args.T // 2, args.T]
        with torch.no_grad():
            for sid in sample_ids:
                sample = test_u[sid : sid + 1].to(device)
                gt_seq = _sample_window(sample, T=args.T, random_t0=False)
                pred_seq = _rollout_2d(enc, dec, heat, gt_seq[..., 0].unsqueeze(-1), args.T, args.dt)

                gt_np = gt_seq.squeeze(0).cpu()
                pred_np = pred_seq.squeeze(0).cpu()

                plot_2d_time_slices(
                    gt=gt_np,
                    pred=pred_np,
                    t_indices=t_indices,
                    out_path_no_ext=os.path.join(image_dir, f"sample_{sid:03d}_slices"),
                    suptitle=f"sample {sid}, full relL2={rel_l2(pred_np, gt_np):.3g}",
                )
                plot_rel_l2_over_time(
                    gt=gt_np,
                    pred=pred_np,
                    out_path_no_ext=os.path.join(image_dir, f"sample_{sid:03d}_relL2_over_time"),
                )
    except Exception as exc:
        print(f"[viz] failed to draw sample figures: {exc}")


if __name__ == "__main__":
    main()
