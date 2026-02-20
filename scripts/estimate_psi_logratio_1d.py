import argparse
import os
import sys

import numpy as np
import torch

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from cli_utils import add_data_mode_args, add_split_args, validate_data_mode_args
from utilities3 import MatReader
from viz_utils import plot_psi_curve


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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Estimate psi(lambda) via log-ratio baseline in 1D.")
    add_data_mode_args(
        parser,
        default_data_mode="single_split",
        default_data_file="data/fractional_diffusion_1d_alpha0.5.mat",
        default_train_file=None,
        default_test_file=None,
    )
    add_split_args(parser, default_train_split=0.8, default_seed=0)

    parser.add_argument("--sub", type=int, default=1)
    parser.add_argument("--sub-t", type=int, default=1)
    parser.add_argument("--split", choices=("all", "train", "test"), default="all")

    parser.add_argument("--amp-threshold", type=float, default=1e-8)
    parser.add_argument("--log-eps", type=float, default=1e-12)
    parser.add_argument("--max-samples", type=int, default=0)

    parser.add_argument("--viz-dir", type=str, default="visualizations/psi_baseline_1d")
    parser.add_argument("--out-npz", type=str, default="")
    parser.add_argument("--plot-psi-true", action="store_true")
    return parser


def _validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    validate_data_mode_args(args, parser)
    if args.sub <= 0:
        parser.error("--sub must be positive.")
    if args.sub_t <= 0:
        parser.error("--sub-t must be positive.")
    if args.amp_threshold < 0:
        parser.error("--amp-threshold must be >= 0.")
    if args.log_eps <= 0:
        parser.error("--log-eps must be positive.")


def _split_indices(total: int, train_split: float, shuffle: bool, seed: int) -> tuple[np.ndarray, np.ndarray]:
    indices = np.arange(total)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)
    split_idx = int(total * train_split)
    return indices[:split_idx], indices[split_idx:]


def _load_single_split(
    args: argparse.Namespace,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float | None]:
    reader = MatReader(args.data_file)
    a = reader.read_field("a")
    u = reader.read_field("u")
    t = reader.read_field("t").reshape(-1)
    alpha = _read_optional_scalar(reader, "alpha")

    if a.ndim != 2 or u.ndim != 3:
        raise ValueError(f"Expected a=(N,S), u=(N,S,T), got {tuple(a.shape)}, {tuple(u.shape)}")

    a = a[:, :: args.sub]
    u = u[:, :: args.sub, :: args.sub_t]
    t = t[:: args.sub_t]

    train_idx, test_idx = _split_indices(a.shape[0], args.train_split, args.shuffle, args.seed)
    if args.split == "train":
        use_idx = train_idx
    elif args.split == "test":
        use_idx = test_idx
    else:
        use_idx = np.concatenate([train_idx, test_idx], axis=0)

    return a[use_idx], u[use_idx], t, alpha


def _load_separate_files(
    args: argparse.Namespace,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float | None]:
    train_reader = MatReader(args.train_file)
    test_reader = MatReader(args.test_file)

    a_train = train_reader.read_field("a")[:, :: args.sub]
    u_train = train_reader.read_field("u")[:, :: args.sub, :: args.sub_t]
    t_train = train_reader.read_field("t").reshape(-1)[:: args.sub_t]

    a_test = test_reader.read_field("a")[:, :: args.sub]
    u_test = test_reader.read_field("u")[:, :: args.sub, :: args.sub_t]
    t_test = test_reader.read_field("t").reshape(-1)[:: args.sub_t]

    if t_train.numel() != t_test.numel() or not torch.allclose(t_train, t_test):
        raise ValueError("Train/test time arrays do not match in separate_files mode.")

    alpha = _read_optional_scalar(train_reader, "alpha")
    if args.split == "train":
        return a_train, u_train, t_train, alpha
    if args.split == "test":
        return a_test, u_test, t_train, alpha
    return torch.cat([a_train, a_test], dim=0), torch.cat([u_train, u_test], dim=0), t_train, alpha


def _maybe_subsample(a: torch.Tensor, u: torch.Tensor, max_samples: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    if max_samples <= 0 or a.shape[0] <= max_samples:
        return a, u
    rng = np.random.default_rng(seed)
    idx = rng.choice(a.shape[0], size=max_samples, replace=False)
    idx = np.sort(idx)
    idx_t = torch.as_tensor(idx, dtype=torch.long)
    return a[idx_t], u[idx_t]


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    _validate_args(args, parser)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.data_mode == "single_split":
        a, u, t, alpha = _load_single_split(args)
    else:
        a, u, t, alpha = _load_separate_files(args)

    a, u = _maybe_subsample(a, u, args.max_samples, args.seed)

    N, S = a.shape
    if u.shape[0] != N or u.shape[1] != S:
        raise ValueError("Shape mismatch in loaded tensors.")

    a_hat = torch.fft.rfft(a, dim=1)  # (N, K)
    u_hat = torch.fft.rfft(u, dim=1)  # (N, K, T)

    abs_a = torch.abs(a_hat)
    abs_u = torch.abs(u_hat)

    y = torch.log(abs_u + args.log_eps) - torch.log(abs_a[..., None] + args.log_eps)
    mask = (abs_a > args.amp_threshold)[..., None] & (abs_u > args.amp_threshold)
    mask_f = mask.to(y.dtype)

    t_vec = t.view(1, 1, -1).to(y.dtype)
    num = torch.sum(mask_f * t_vec * y, dim=(0, 2))
    den = torch.sum(mask_f * (t_vec**2), dim=(0, 2))

    tiny = torch.finfo(y.dtype).tiny
    psi_hat = torch.clamp(-num / (den + tiny), min=0.0)
    psi_hat[0] = 0.0

    counts = torch.sum(mask_f, dim=(0, 2))

    k = torch.fft.rfftfreq(S, d=1.0 / S)
    lam = (2.0 * np.pi * k) ** 2

    psi_true = None
    if args.plot_psi_true and alpha is not None:
        psi_true = torch.pow(lam, alpha)

    os.makedirs(args.viz_dir, exist_ok=True)
    plot_psi_curve(
        lam=lam,
        psi_pred=psi_hat,
        psi_true=psi_true,
        logx=True,
        logy=True,
        title=f"psi baseline (split={args.split}, N={N})",
        out_path_no_ext=os.path.join(args.viz_dir, "psi_baseline_curve"),
    )

    out_npz = args.out_npz if args.out_npz else os.path.join(args.viz_dir, "psi_baseline.npz")
    out_npz_dir = os.path.dirname(out_npz)
    if out_npz_dir:
        os.makedirs(out_npz_dir, exist_ok=True)
    np.savez(
        out_npz,
        lam=lam.detach().cpu().numpy(),
        psi_hat=psi_hat.detach().cpu().numpy(),
        counts=counts.detach().cpu().numpy(),
        t=t.detach().cpu().numpy(),
        alpha=np.array(np.nan if alpha is None else alpha, dtype=np.float32),
    )

    print(f"Saved: {out_npz}")
    print(f"Saved psi baseline plot to: {os.path.join(args.viz_dir, 'psi_baseline_curve')}.*")


if __name__ == "__main__":
    main()
