"""Fourier Neural Operator for 1D time-series rollout tasks."""

from __future__ import annotations

import argparse
import os
from timeit import default_timer

import torch
import torch.nn.functional as F

from cli_utils import add_data_mode_args, add_split_args, validate_data_mode_args
from utilities3 import *
from viz_utils import (
    LearningCurve,
    plot_1d_prediction,
    plot_error_histogram,
    plot_learning_curve,
    plot_rel_l2_over_time_1d,
    rel_l2,
)


torch.manual_seed(0)
np.random.seed(0)


class TimeGaussianNormalizer1D:
    """Time-independent per-space normalization for [N,S,T]."""

    def __init__(self, u_train: torch.Tensor, eps: float = 1e-5):
        self.eps = eps
        self.mean = torch.mean(u_train, dim=(0, 2), keepdim=True)
        self.std = torch.std(u_train, dim=(0, 2), keepdim=True)

    def encode(self, u: torch.Tensor) -> torch.Tensor:
        return (u - self.mean) / (self.std + self.eps)

    def decode(self, u: torch.Tensor) -> torch.Tensor:
        return u * (self.std + self.eps) + self.mean

    def to(self, device: torch.device) -> "TimeGaussianNormalizer1D":
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        return self


class SpectralConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, modes1: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.scale = 1.0 / max(1, in_channels * out_channels)
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat)
        )

    def compl_mul1d(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bix,iox->box", x, w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        x_ft = torch.fft.rfft(x)
        out_ft = torch.zeros(
            bsz,
            self.out_channels,
            x.size(-1) // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )
        out_ft[:, :, : self.modes1] = self.compl_mul1d(x_ft[:, :, : self.modes1], self.weights1)
        return torch.fft.irfft(out_ft, n=x.size(-1))


class MLP(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, mid_channels: int):
        super().__init__()
        self.mlp1 = nn.Conv1d(in_channels, mid_channels, 1)
        self.mlp2 = nn.Conv1d(mid_channels, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.gelu(self.mlp1(x))
        return self.mlp2(x)


class FNO1dTime(nn.Module):
    def __init__(self, modes: int, width: int, in_channels: int):
        super().__init__()
        self.modes1 = modes
        self.width = width
        self.p = nn.Linear(in_channels, self.width)  # input: T_in + x-coordinate

        self.conv0 = SpectralConv1d(width, width, modes)
        self.conv1 = SpectralConv1d(width, width, modes)
        self.conv2 = SpectralConv1d(width, width, modes)
        self.conv3 = SpectralConv1d(width, width, modes)

        self.mlp0 = MLP(width, width, width)
        self.mlp1 = MLP(width, width, width)
        self.mlp2 = MLP(width, width, width)
        self.mlp3 = MLP(width, width, width)

        self.w0 = nn.Conv1d(width, width, 1)
        self.w1 = nn.Conv1d(width, width, 1)
        self.w2 = nn.Conv1d(width, width, 1)
        self.w3 = nn.Conv1d(width, width, 1)

        self.q = MLP(width, 1, width * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.p(x)
        x = x.permute(0, 2, 1)

        x1 = self.mlp0(self.conv0(x))
        x = F.gelu(x1 + self.w0(x))

        x1 = self.mlp1(self.conv1(x))
        x = F.gelu(x1 + self.w1(x))

        x1 = self.mlp2(self.conv2(x))
        x = F.gelu(x1 + self.w2(x))

        x1 = self.mlp3(self.conv3(x))
        x = x1 + self.w3(x)

        x = self.q(x)
        return x.permute(0, 2, 1)

    @staticmethod
    def get_grid(shape: torch.Size, device: torch.device) -> torch.Tensor:
        bsz, size_x = shape[0], shape[1]
        gridx = torch.linspace(0.0, 1.0, size_x, dtype=torch.float32, device=device)
        return gridx.view(1, size_x, 1).repeat(bsz, 1, 1)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fourier Neural Operator 1D Time")
    add_data_mode_args(
        parser,
        default_data_mode="single_split",
        default_data_file="data/burgers_1d_ts.mat",
        default_train_file=None,
        default_test_file=None,
    )
    add_split_args(parser, default_train_split=0.8, default_seed=0)

    parser.add_argument("--field", type=str, default="u", help="MAT field name for time-series data")
    parser.add_argument("--ntrain", type=int, default=1000)
    parser.add_argument("--ntest", type=int, default=100)
    parser.add_argument("--sub", type=int, default=1)
    parser.add_argument("--S", type=int, default=1024)
    parser.add_argument("--normalize", choices=("none", "unit_gaussian"), default="unit_gaussian")
    parser.add_argument("--T-in", type=int, default=1)
    parser.add_argument("--T", type=int, default=40)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--modes", type=int, default=16)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser


def _validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    validate_data_mode_args(args, parser)
    if args.ntrain <= 0 or args.ntest <= 0:
        parser.error("--ntrain and --ntest must be positive")
    if args.sub <= 0:
        parser.error("--sub must be positive")
    if args.S <= 1:
        parser.error("--S must be > 1")
    if args.T_in <= 0 or args.T <= 0 or args.step <= 0:
        parser.error("--T-in, --T, --step must be positive")


def _pick_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("--device=cuda requested but CUDA is unavailable")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _to_time_last_1d(u: torch.Tensor, expected_s: int) -> torch.Tensor:
    if u.ndim != 3:
        raise ValueError(f"Expected 3D tensor for 1D time-series, got shape={tuple(u.shape)}")
    if u.shape[1] == expected_s:
        return u
    if u.shape[2] == expected_s:
        return u.transpose(1, 2)
    raise ValueError(f"Could not infer spatial axis for shape={tuple(u.shape)}, expected spatial={expected_s}")


def _read_u(path: str, field: str) -> torch.Tensor:
    reader = MatReader(path)
    return reader.read_field(field).float()


def _split_indices(total: int, train_split: float, shuffle: bool, seed: int) -> tuple[np.ndarray, np.ndarray]:
    idx = np.arange(total)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(idx)
    split = int(total * train_split)
    return idx[:split], idx[split:]


def _prepare_data(args: argparse.Namespace) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    def _slice_traj(u: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        s = u.shape[1]
        t_total = u.shape[2]
        if args.T_in + args.T > t_total:
            raise ValueError(f"Need T_in+T <= T_total, got {args.T_in}+{args.T} > {t_total}")
        a = u[:, :, : args.T_in]
        y = u[:, :, args.T_in : args.T_in + args.T]
        if s != args.S:
            raise ValueError(f"Expected S={args.S}, got {s}. Adjust --S/--sub.")
        return a, y

    if args.data_mode == "single_split":
        u_raw = _read_u(args.data_file, args.field)
        u_all = _to_time_last_1d(u_raw, expected_s=args.S * args.sub)
        u_all = u_all[:, :: args.sub, :]
        total = u_all.shape[0]
        train_idx, test_idx = _split_indices(total, args.train_split, args.shuffle, args.seed)
        if args.ntrain > len(train_idx) or args.ntest > len(test_idx):
            raise ValueError(
                f"Not enough trajectories: total={total}, available_train={len(train_idx)}, "
                f"available_test={len(test_idx)}, requested ntrain={args.ntrain}, ntest={args.ntest}"
            )
        u_train = u_all[train_idx[: args.ntrain]]
        u_test = u_all[test_idx[: args.ntest]]
    else:
        u_train_raw = _read_u(args.train_file, args.field)
        u_test_raw = _read_u(args.test_file, args.field)
        u_train = _to_time_last_1d(u_train_raw, expected_s=args.S * args.sub)[:, :: args.sub, :]
        u_test = _to_time_last_1d(u_test_raw, expected_s=args.S * args.sub)[:, :: args.sub, :]
        if args.ntrain > u_train.shape[0] or args.ntest > u_test.shape[0]:
            raise ValueError(
                f"Not enough trajectories in separate_files mode: train={u_train.shape[0]}, test={u_test.shape[0]}, "
                f"requested ntrain={args.ntrain}, ntest={args.ntest}"
            )
        u_train = u_train[: args.ntrain]
        u_test = u_test[: args.ntest]

    train_a, train_u = _slice_traj(u_train)
    test_a, test_u = _slice_traj(u_test)
    return train_a, train_u, test_a, test_u


def _rollout_chunked(model: nn.Module, xx: torch.Tensor, horizon: int, step: int) -> torch.Tensor:
    preds = []
    state = xx
    t = 0
    while t < horizon:
        chunk = min(step, horizon - t)
        chunk_preds = []
        for _ in range(chunk):
            im = model(state)
            chunk_preds.append(im)
            state = torch.cat((state[..., 1:], im), dim=-1)
        preds.append(torch.cat(chunk_preds, dim=-1))
        t += chunk
    return torch.cat(preds, dim=-1)


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    _validate_args(args, parser)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = _pick_device(args.device)

    train_a, train_u, test_a, test_u = _prepare_data(args)
    normalizer: TimeGaussianNormalizer1D | None = None
    if args.normalize == "unit_gaussian":
        full_train = torch.cat([train_a, train_u], dim=-1)
        normalizer = TimeGaussianNormalizer1D(full_train)
        train_a = normalizer.encode(train_a)
        train_u = normalizer.encode(train_u)
        test_a = normalizer.encode(test_a)
        test_u = normalizer.encode(test_u)

    ntrain, s, _ = train_a.shape
    ntest = test_a.shape[0]
    print(f"[info] device={device}, train={ntrain}, test={ntest}, S={s}, T_in={args.T_in}, T={args.T}")
    id_baseline = rel_l2(test_a[:, :, -1:], test_u[:, :, :1])
    print(f"[diag] persistence baseline relL2 (t+1 <- last input) = {id_baseline:.6f}")

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(train_a, train_u), batch_size=args.batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(test_a, test_u), batch_size=args.batch_size, shuffle=False
    )

    model = FNO1dTime(args.modes, args.width, in_channels=args.T_in + 1).to(device)
    print(count_params(model))

    iterations = max(1, args.epochs * max(1, ntrain // args.batch_size))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iterations)
    myloss = LpLoss(size_average=False)

    viz_dir = os.path.join("visualizations", "fourier_1d_time")
    os.makedirs(viz_dir, exist_ok=True)

    hist_epochs: list[int] = []
    hist_train_step: list[float] = []
    hist_train_full: list[float] = []
    hist_test_step: list[float] = []
    hist_test_full: list[float] = []

    for ep in range(args.epochs):
        model.train()
        t1 = default_timer()
        train_l2_step = 0.0
        train_l2_full = 0.0

        for xx, yy in train_loader:
            xx = xx.to(device)
            yy = yy.to(device)
            pred = _rollout_chunked(model, xx, horizon=args.T, step=args.step)
            bs = xx.shape[0]

            loss_step = 0.0
            for t in range(0, args.T, args.step):
                h = min(args.step, args.T - t)
                loss_step = loss_step + myloss(pred[..., t : t + h].reshape(bs, -1), yy[..., t : t + h].reshape(bs, -1))

            train_l2_step += float(loss_step.item())
            train_l2_full += float(myloss(pred.reshape(bs, -1), yy.reshape(bs, -1)).item())

            optimizer.zero_grad()
            loss_step.backward()
            optimizer.step()
            scheduler.step()

        model.eval()
        test_l2_step = 0.0
        test_l2_full = 0.0
        with torch.no_grad():
            for xx, yy in test_loader:
                xx = xx.to(device)
                yy = yy.to(device)
                pred = _rollout_chunked(model, xx, horizon=args.T, step=args.step)
                bs = xx.shape[0]

                loss_step = 0.0
                for t in range(0, args.T, args.step):
                    h = min(args.step, args.T - t)
                    loss_step = loss_step + myloss(pred[..., t : t + h].reshape(bs, -1), yy[..., t : t + h].reshape(bs, -1))

                test_l2_step += float(loss_step.item())
                test_l2_full += float(myloss(pred.reshape(bs, -1), yy.reshape(bs, -1)).item())

        t2 = default_timer()
        tr_step = train_l2_step / ntrain / (args.T / args.step)
        tr_full = train_l2_full / ntrain
        te_step = test_l2_step / ntest / (args.T / args.step)
        te_full = test_l2_full / ntest
        print(ep, t2 - t1, tr_step, tr_full, te_step, te_full)

        hist_epochs.append(ep)
        hist_train_step.append(tr_step)
        hist_train_full.append(tr_full)
        hist_test_step.append(te_step)
        hist_test_full.append(te_full)

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
            title="fourier_1d_time: stepwise relative L2",
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
            title="fourier_1d_time: full-trajectory relative L2",
        )

        test_a_dev = test_a.to(device)
        test_u_cpu = test_u.cpu()
        n_hist = min(ntest, 50)
        per_sample_full: list[float] = []
        with torch.no_grad():
            for i in range(n_hist):
                pred_i = _rollout_chunked(model, test_a_dev[i : i + 1], horizon=args.T, step=args.step).squeeze(0).cpu()
                gt_i = test_u_cpu[i]
                if normalizer is not None:
                    gt_i = normalizer.decode(gt_i.unsqueeze(0)).squeeze(0)
                    pred_i = normalizer.decode(pred_i.unsqueeze(0)).squeeze(0)
                per_sample_full.append(rel_l2(pred_i, gt_i))
        plot_error_histogram(
            per_sample_full,
            os.path.join(viz_dir, f"test_full_relL2_hist_first{n_hist}"),
            title=f"full relL2 histogram (first {n_hist} test trajectories)",
        )

        sample_ids = [0, min(1, ntest - 1), min(2, ntest - 1)]
        t_ids = [0, args.T // 2, args.T - 1]
        x_grid = np.linspace(0.0, 1.0, s)

        with torch.no_grad():
            for i in sample_ids:
                pred_i = _rollout_chunked(model, test_a_dev[i : i + 1], horizon=args.T, step=args.step).squeeze(0).cpu()
                gt_i = test_u_cpu[i]
                input_i = test_a[i, :, 0]
                if normalizer is not None:
                    gt_i = normalizer.decode(gt_i.unsqueeze(0)).squeeze(0)
                    pred_i = normalizer.decode(pred_i.unsqueeze(0)).squeeze(0)
                    input_i = normalizer.decode(input_i.view(1, -1, 1)).view(-1)
                plot_rel_l2_over_time_1d(
                    gt=gt_i,
                    pred=pred_i,
                    out_path_no_ext=os.path.join(viz_dir, f"sample_{i:03d}_relL2_over_time"),
                )
                for tt in t_ids:
                    plot_1d_prediction(
                        x=x_grid,
                        gt=gt_i[:, tt],
                        pred=pred_i[:, tt],
                        input_u0=input_i,
                        out_path_no_ext=os.path.join(viz_dir, f"sample_{i:03d}_t{tt:03d}"),
                        title_prefix=f"sample {i}, t={tt}: ",
                    )
    except Exception as e:  # pragma: no cover
        print(f"[viz] failed: {e}")


if __name__ == "__main__":
    main()
