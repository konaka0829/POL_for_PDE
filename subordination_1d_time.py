import argparse
import os
from timeit import default_timer

import numpy as np
import torch
import torch.nn as nn

from cli_utils import add_data_mode_args, add_split_args, validate_data_mode_args
from utilities3 import MatReader, count_params
from viz_utils import (
    LearningCurve,
    plot_1d_prediction,
    plot_1d_prediction_multi,
    plot_error_histogram,
    plot_learning_curve,
    plot_psi_curve,
    rel_l2,
)


def _relative_l2_batch(pred: torch.Tensor, gt: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    pred_f = pred.reshape(pred.shape[0], -1)
    gt_f = gt.reshape(gt.shape[0], -1)
    num = torch.linalg.norm(pred_f - gt_f, dim=1)
    den = torch.linalg.norm(gt_f, dim=1)
    return torch.mean(num / (den + eps))


def _read_optional_scalar(reader: MatReader, field: str) -> float | None:
    try:
        value = reader.read_field(field)
    except Exception:
        return None
    if torch.is_tensor(value):
        if value.numel() == 1:
            return float(value.reshape(-1)[0].item())
        return None
    arr = np.asarray(value)
    if arr.size == 1:
        return float(arr.reshape(-1)[0])
    return None


class BernsteinPsi(nn.Module):
    def __init__(
        self,
        J: int,
        s_min: float = 1e-3,
        s_max: float = 1e3,
        learn_s: bool = False,
        s_eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if J <= 0:
            raise ValueError("J must be positive.")
        if s_min <= 0.0 or s_max <= 0.0:
            raise ValueError("s_min and s_max must be positive.")
        if s_max < s_min:
            raise ValueError("s_max must be >= s_min.")

        self.s_eps = float(s_eps)
        self.learn_s = bool(learn_s)

        self.log_a = nn.Parameter(torch.tensor(-10.0))
        self.log_b = nn.Parameter(torch.tensor(-10.0))
        self.log_alpha = nn.Parameter(torch.zeros(J))

        s0 = torch.logspace(np.log10(s_min), np.log10(s_max), J)
        theta_s0 = torch.log(torch.clamp(s0, min=1e-12))
        if self.learn_s:
            self.theta_s = nn.Parameter(theta_s0.clone())
        else:
            self.register_buffer("theta_s", theta_s0)

    def positive_params(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        a0 = torch.nn.functional.softplus(self.log_a)
        b = torch.nn.functional.softplus(self.log_b)
        alpha = torch.nn.functional.softplus(self.log_alpha)
        s = torch.exp(self.theta_s) + self.s_eps
        return a0, b, alpha, s

    def forward(self, lam: torch.Tensor) -> torch.Tensor:
        # lam: (..., K) or (K,)
        lam = torch.clamp(lam, min=0.0)
        a0, b, alpha, s = self.positive_params()
        atoms = 1.0 - torch.exp(-lam[..., None] * s[None, ...])
        return a0 + b * lam + torch.sum(alpha[None, ...] * atoms, dim=-1)


class SubordinatedHeatOperator1D(nn.Module):
    def __init__(self, S: int, psi: BernsteinPsi) -> None:
        super().__init__()
        self.S = int(S)
        self.psi = psi
        k = torch.fft.rfftfreq(self.S, d=1.0 / self.S)
        lam = (2.0 * np.pi * k) ** 2
        self.register_buffer("lam", lam)

    def forward(self, a: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # a: (B, S) or (B, S, 1), t: (T,), out: (B, S, T)
        if a.ndim == 3 and a.shape[-1] == 1:
            a = a[..., 0]
        if a.ndim != 2:
            raise ValueError(f"Expected a shape (B,S) or (B,S,1), got {tuple(a.shape)}")
        if t.ndim != 1:
            raise ValueError(f"Expected t shape (T,), got {tuple(t.shape)}")

        a_hat = torch.fft.rfft(a, dim=-1)  # (B, K), complex
        psi_lam = torch.clamp(self.psi(self.lam), min=0.0)  # (K,)
        decay = torch.exp(-t[:, None] * psi_lam[None, :])  # (T, K), real
        u_hat = a_hat[:, None, :] * decay[None, :, :]  # (B, T, K), complex
        u = torch.fft.irfft(u_hat, n=self.S, dim=-1)  # (B, T, S), real
        return u.permute(0, 2, 1).contiguous()  # (B, S, T)

    def forward_mc(
        self,
        a: torch.Tensor,
        t: torch.Tensor,
        mc_samples: int,
        generator: torch.Generator | None = None,
        chunk_size: int | None = None,
    ) -> torch.Tensor:
        if mc_samples <= 0:
            raise ValueError("mc_samples must be positive.")
        if a.ndim == 3 and a.shape[-1] == 1:
            a = a[..., 0]
        if a.ndim != 2:
            raise ValueError(f"Expected a shape (B,S) or (B,S,1), got {tuple(a.shape)}")
        if t.ndim != 1:
            raise ValueError(f"Expected t shape (T,), got {tuple(t.shape)}")

        a_hat = torch.fft.rfft(a, dim=-1)  # (B, K), complex
        B, K = a_hat.shape
        T = t.numel()
        lam = self.lam
        a0, b, alpha, s = self.psi.positive_params()

        if chunk_size is None or chunk_size <= 0:
            chunk_size = mc_samples
        chunk_size = min(chunk_size, mc_samples)

        out = torch.empty((B, self.S, T), device=a.device, dtype=a.dtype)
        for j in range(T):
            t_j = t[j]
            rates_j = torch.clamp(alpha * t_j, min=0.0)  # (J,)
            u_hat_sum = torch.zeros((B, K), device=a_hat.device, dtype=a_hat.dtype)
            used = 0
            while used < mc_samples:
                m = min(chunk_size, mc_samples - used)
                rate_tensor = rates_j.view(1, 1, -1).expand(m, B, -1)
                N = torch.poisson(rate_tensor, generator=generator)
                tau = b * t_j + torch.sum(N * s.view(1, 1, -1), dim=-1)  # (m, B)
                mult = torch.exp(-tau[..., None] * lam.view(1, 1, -1))  # (m, B, K)
                u_hat_samples = a_hat[None, :, :] * mult
                u_hat_sum = u_hat_sum + u_hat_samples.sum(dim=0)
                used += m

            u_hat_mean = (u_hat_sum / float(mc_samples)) * torch.exp(-a0 * t_j)
            u_j = torch.fft.irfft(u_hat_mean, n=self.S, dim=-1)
            out[:, :, j] = u_j
        return out


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="1D subordination-based operator learning")
    add_data_mode_args(
        parser,
        default_data_mode="single_split",
        default_data_file="data/fractional_diffusion_1d_alpha0.5.mat",
        default_train_file=None,
        default_test_file=None,
    )
    add_split_args(parser, default_train_split=0.8, default_seed=0)

    parser.add_argument("--ntrain", type=int, default=1000)
    parser.add_argument("--ntest", type=int, default=200)
    parser.add_argument("--sub", type=int, default=1)
    parser.add_argument("--sub-t", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--epochs", type=int, default=300)

    parser.add_argument("--psi-J", type=int, default=32)
    parser.add_argument("--learn-s", action="store_true")
    parser.add_argument("--psi-s-min", type=float, default=1e-3)
    parser.add_argument("--psi-s-max", type=float, default=1e3)
    parser.add_argument("--psi-eps", type=float, default=1e-8)

    parser.add_argument("--viz-dir", type=str, default="visualizations/subordination_1d_time")
    parser.add_argument("--plot-psi", action="store_true")
    parser.add_argument("--plot-samples", type=int, default=3)
    parser.add_argument("--mc-samples", type=int, default=0)
    parser.add_argument("--mc-seed", type=int, default=0)
    parser.add_argument("--mc-batch-size", type=int, default=0)
    parser.add_argument("--mc-chunk", type=int, default=0)
    return parser


def _validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    validate_data_mode_args(args, parser)
    if args.sub <= 0:
        parser.error("--sub must be positive.")
    if args.sub_t <= 0:
        parser.error("--sub-t must be positive.")
    if args.ntrain <= 0 or args.ntest <= 0:
        parser.error("--ntrain and --ntest must be positive.")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive.")
    if args.epochs <= 0:
        parser.error("--epochs must be positive.")
    if args.learning_rate <= 0:
        parser.error("--learning-rate must be positive.")
    if args.mc_samples < 0:
        parser.error("--mc-samples must be >= 0.")
    if args.mc_batch_size < 0:
        parser.error("--mc-batch-size must be >= 0.")
    if args.mc_chunk < 0:
        parser.error("--mc-chunk must be >= 0.")


def _load_single_split(
    args: argparse.Namespace,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float | None]:
    reader = MatReader(args.data_file)
    a_data = reader.read_field("a")
    u_data = reader.read_field("u")
    t = reader.read_field("t").reshape(-1)
    alpha = _read_optional_scalar(reader, "alpha")

    if a_data.ndim != 2 or u_data.ndim != 3:
        raise ValueError(f"Expected a=(N,S), u=(N,S,T), got {tuple(a_data.shape)}, {tuple(u_data.shape)}")
    if t.ndim != 1:
        raise ValueError(f"Expected t=(T,), got {tuple(t.shape)}")

    a_data = a_data[:, :: args.sub]
    u_data = u_data[:, :: args.sub, :: args.sub_t]
    t = t[:: args.sub_t]

    total = a_data.shape[0]
    indices = np.arange(total)
    if args.shuffle:
        np.random.shuffle(indices)
    split_idx = int(total * args.train_split)
    train_idx = indices[:split_idx]
    test_idx = indices[split_idx:]

    if args.ntrain > len(train_idx) or args.ntest > len(test_idx):
        raise ValueError(
            f"Not enough samples for ntrain={args.ntrain}, ntest={args.ntest} "
            f"with train_split={args.train_split} and total={total}."
        )

    train_idx = train_idx[: args.ntrain]
    test_idx = test_idx[: args.ntest]
    return (
        a_data[train_idx],
        u_data[train_idx],
        a_data[test_idx],
        u_data[test_idx],
        t,
        alpha,
    )


def _load_separate_files(
    args: argparse.Namespace,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float | None]:
    train_reader = MatReader(args.train_file)
    test_reader = MatReader(args.test_file)

    a_train_all = train_reader.read_field("a")[:, :: args.sub]
    u_train_all = train_reader.read_field("u")[:, :: args.sub, :: args.sub_t]
    t_train = train_reader.read_field("t").reshape(-1)[:: args.sub_t]

    a_test_all = test_reader.read_field("a")[:, :: args.sub]
    u_test_all = test_reader.read_field("u")[:, :: args.sub, :: args.sub_t]
    t_test = test_reader.read_field("t").reshape(-1)[:: args.sub_t]

    if args.ntrain > a_train_all.shape[0]:
        raise ValueError(f"ntrain={args.ntrain} exceeds available train samples={a_train_all.shape[0]}.")
    if args.ntest > a_test_all.shape[0]:
        raise ValueError(f"ntest={args.ntest} exceeds available test samples={a_test_all.shape[0]}.")
    if t_train.numel() != t_test.numel() or not torch.allclose(t_train, t_test):
        raise ValueError("Train/test time arrays do not match in separate_files mode.")

    alpha = _read_optional_scalar(train_reader, "alpha")
    return (
        a_train_all[: args.ntrain],
        u_train_all[: args.ntrain],
        a_test_all[: args.ntest],
        u_test_all[: args.ntest],
        t_train,
        alpha,
    )


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    _validate_args(args, parser)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.data_mode == "single_split":
        x_train, y_train, x_test, y_test, t, alpha = _load_single_split(args)
    else:
        x_train, y_train, x_test, y_test, t, alpha = _load_separate_files(args)

    S = int(x_train.shape[1])
    T = int(y_train.shape[-1])
    if y_train.shape[1] != S:
        raise ValueError("Shape mismatch: x_train and y_train spatial dimension differ.")
    if t.numel() != T:
        raise ValueError(f"Time dimension mismatch: y has T={T} but t has {t.numel()}.")

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_train, y_train),
        batch_size=args.batch_size,
        shuffle=True,
    )
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_test, y_test),
        batch_size=args.batch_size,
        shuffle=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    psi = BernsteinPsi(
        J=args.psi_J,
        s_min=args.psi_s_min,
        s_max=args.psi_s_max,
        learn_s=args.learn_s,
        s_eps=args.psi_eps,
    )
    model = SubordinatedHeatOperator1D(S=S, psi=psi).to(device)
    t_dev = t.to(device)
    print(f"device={device}, parameters={count_params(model)}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    hist_epochs: list[int] = []
    hist_train_rel_l2: list[float] = []
    hist_test_rel_l2: list[float] = []

    os.makedirs(args.viz_dir, exist_ok=True)

    for ep in range(args.epochs):
        t0 = default_timer()
        model.train()
        train_sum = 0.0
        train_count = 0
        for a_batch, u_batch in train_loader:
            a_batch = a_batch.to(device)
            u_batch = u_batch.to(device)

            optimizer.zero_grad()
            pred = model(a_batch, t_dev)
            loss = _relative_l2_batch(pred, u_batch)
            loss.backward()
            optimizer.step()

            bs = a_batch.shape[0]
            train_sum += float(loss.item()) * bs
            train_count += bs

        model.eval()
        test_sum = 0.0
        test_count = 0
        with torch.no_grad():
            for a_batch, u_batch in test_loader:
                a_batch = a_batch.to(device)
                u_batch = u_batch.to(device)
                pred = model(a_batch, t_dev)
                loss = _relative_l2_batch(pred, u_batch)
                bs = a_batch.shape[0]
                test_sum += float(loss.item()) * bs
                test_count += bs

        train_rel = train_sum / max(train_count, 1)
        test_rel = test_sum / max(test_count, 1)

        hist_epochs.append(ep)
        hist_train_rel_l2.append(train_rel)
        hist_test_rel_l2.append(test_rel)

        t1 = default_timer()
        print(f"ep={ep:04d} time={t1-t0:.2f}s train_relL2={train_rel:.6f} test_relL2={test_rel:.6f}")

    # Learning curve
    plot_learning_curve(
        LearningCurve(
            epochs=hist_epochs,
            train=hist_train_rel_l2,
            test=hist_test_rel_l2,
            train_label="train (relL2)",
            test_label="test (relL2)",
            metric_name="relative L2",
        ),
        out_path_no_ext=os.path.join(args.viz_dir, "learning_curve"),
        logy=True,
        title="subordination_1d_time",
    )

    # Test predictions and histogram
    model.eval()
    with torch.no_grad():
        pred_test_det = model(x_test.to(device), t_dev).cpu()
    per_sample_err = [rel_l2(pred_test_det[i], y_test[i]) for i in range(pred_test_det.shape[0])]
    plot_error_histogram(per_sample_err, os.path.join(args.viz_dir, "error_hist"))

    pred_test_mc = None
    if args.mc_samples > 0:
        mc_bs = args.batch_size if args.mc_batch_size <= 0 else args.mc_batch_size
        mc_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(x_test, y_test),
            batch_size=mc_bs,
            shuffle=False,
        )
        gen_device = "cuda" if device.type == "cuda" else "cpu"
        mc_generator = torch.Generator(device=gen_device)
        mc_generator.manual_seed(args.mc_seed)
        mc_batches: list[torch.Tensor] = []
        with torch.no_grad():
            for a_batch, _ in mc_loader:
                a_batch = a_batch.to(device)
                pred_batch = model.forward_mc(
                    a_batch,
                    t_dev,
                    mc_samples=args.mc_samples,
                    generator=mc_generator,
                    chunk_size=args.mc_chunk if args.mc_chunk > 0 else None,
                )
                mc_batches.append(pred_batch.cpu())
        pred_test_mc = torch.cat(mc_batches, dim=0)

        err_mc_vs_det = [rel_l2(pred_test_mc[i], pred_test_det[i]) for i in range(pred_test_det.shape[0])]
        plot_error_histogram(err_mc_vs_det, os.path.join(args.viz_dir, "mc_vs_det_hist"))
        err_mc_vs_gt = [rel_l2(pred_test_mc[i], y_test[i]) for i in range(pred_test_det.shape[0])]
        plot_error_histogram(err_mc_vs_gt, os.path.join(args.viz_dir, "mc_vs_gt_hist"))

    # Representative sample plots at t = 0, T//2, T-1
    sample_count = max(1, min(args.plot_samples, x_test.shape[0]))
    sample_ids = list(range(sample_count))
    t_indices = sorted(set([0, T // 2, T - 1]))
    x_grid = np.linspace(0.0, 1.0, S, endpoint=False)
    for sid in sample_ids:
        for tidx in t_indices:
            out_base = os.path.join(args.viz_dir, f"sample_{sid:03d}_t{tidx:03d}")
            if pred_test_mc is None:
                plot_1d_prediction(
                    x=x_grid,
                    gt=y_test[sid, :, tidx],
                    pred=pred_test_det[sid, :, tidx],
                    input_u0=x_test[sid],
                    out_path_no_ext=out_base,
                    title_prefix=f"sample={sid}, t_idx={tidx}:",
                )
            else:
                plot_1d_prediction_multi(
                    x=x_grid,
                    gt=y_test[sid, :, tidx],
                    preds={
                        "deterministic": pred_test_det[sid, :, tidx],
                        "mc": pred_test_mc[sid, :, tidx],
                    },
                    input_u0=x_test[sid],
                    out_path_no_ext=out_base,
                    title_prefix=f"sample={sid}, t_idx={tidx}:",
                )

    if args.plot_psi:
        lam = model.lam.detach().cpu()
        psi_pred = model.psi(lam.to(device)).detach().cpu()
        psi_true = None
        if alpha is not None:
            psi_true = torch.pow(lam, alpha)
        plot_psi_curve(
            lam=lam,
            psi_pred=psi_pred,
            psi_true=psi_true,
            logx=True,
            logy=True,
            title="learned psi vs true psi",
            out_path_no_ext=os.path.join(args.viz_dir, "psi_curve"),
        )

    print(f"Saved visualizations to: {args.viz_dir}")


if __name__ == "__main__":
    main()
