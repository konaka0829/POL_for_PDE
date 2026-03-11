"""
Digital-only PI-SC-OF-RNO for 2D Darcy flow.

Phase 0 implementation:
- Fixed random optical Fourier reservoir features (OF-RNO style)
- Readout-only learning
- Supervised and label-free (PDE residual) fitting modes
"""

import argparse
import os
import sys
import types
from timeit import default_timer

import numpy as np
import torch
import torch.nn as nn

# utilities3 imports h5py unconditionally. Provide a lightweight stub when h5py
# is unavailable so smoke/help can run in minimal environments.
try:
    import h5py  # noqa: F401
except ModuleNotFoundError:
    h5py_stub = types.ModuleType("h5py")

    def _missing_h5py(*_args, **_kwargs):
        raise ModuleNotFoundError("h5py is required to read HDF5-based .mat files")

    h5py_stub.File = _missing_h5py
    sys.modules["h5py"] = h5py_stub

from utilities3 import MatReader, UnitGaussianNormalizer, LpLoss
from cli_utils import add_data_mode_args, validate_data_mode_args
from viz_utils import (
    LearningCurve,
    plot_2d_comparison,
    plot_error_histogram,
    plot_learning_curve,
    rel_l2,
)


torch.manual_seed(0)
np.random.seed(0)


# ------------------------------------------------------------
# Utility helpers
# ------------------------------------------------------------
def _activation(x: torch.Tensor, name: str) -> torch.Tensor:
    if name == "tanh":
        return torch.tanh(x)
    if name == "gelu":
        return torch.nn.functional.gelu(x)
    if name == "relu":
        return torch.relu(x)
    if name == "identity":
        return x
    raise ValueError(f"unknown nonlinearity: {name}")


def _build_grid(batch: int, size_x: int, size_y: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    gridx = torch.linspace(0.0, 1.0, size_x, device=device, dtype=dtype).view(1, size_x, 1, 1)
    gridx = gridx.repeat(batch, 1, size_y, 1)
    gridy = torch.linspace(0.0, 1.0, size_y, device=device, dtype=dtype).view(1, 1, size_y, 1)
    gridy = gridy.repeat(batch, size_x, 1, 1)
    return torch.cat((gridx, gridy), dim=-1)


def _build_lowfreq_mask(s: int, k: int) -> torch.Tensor:
    idx = torch.arange(s)
    signed = torch.where(idx <= (s // 2), idx, idx - s)
    mask = (signed.abs() <= k)
    return (mask[:, None] & mask[None, :]).to(torch.float32)


def _boundary_mask(s: int, device: torch.device) -> torch.Tensor:
    mask = torch.zeros(s, s, dtype=torch.bool, device=device)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _file_missing(args: argparse.Namespace) -> bool:
    if args.data_mode == "single_split":
        return (not args.data_file) or (not os.path.exists(args.data_file))
    return (
        (not args.train_file)
        or (not os.path.exists(args.train_file))
        or (not args.test_file)
        or (not os.path.exists(args.test_file))
    )


def _generate_smoke_data(ntrain: int, ntest: int, s: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    ntrain_eff = max(4, min(ntrain, 32))
    ntest_eff = max(2, min(ntest, 8))
    s_eff = max(8, min(s, 17))

    x = torch.linspace(0.0, 1.0, s_eff)
    y = torch.linspace(0.0, 1.0, s_eff)
    xx, yy = torch.meshgrid(x, y, indexing="ij")

    coeff_all = []
    sol_all = []
    total = ntrain_eff + ntest_eff
    for i in range(total):
        amp = 0.1 + 0.05 * (i % 5)
        phase = 0.3 * (i + 1)
        coeff = 0.8 + amp * torch.sin(2.0 * np.pi * (xx + phase)) * torch.cos(2.0 * np.pi * (yy - phase))
        coeff = coeff + 0.05 * torch.randn_like(coeff)
        coeff = torch.clamp(coeff, min=0.05)

        # Smooth synthetic target with zero boundary.
        sol = torch.sin(np.pi * xx) * torch.sin(np.pi * yy) / (coeff.mean() + 0.2)
        sol = sol + 0.02 * torch.randn_like(sol)
        sol[0, :] = 0.0
        sol[-1, :] = 0.0
        sol[:, 0] = 0.0
        sol[:, -1] = 0.0

        coeff_all.append(coeff)
        sol_all.append(sol)

    coeff_all = torch.stack(coeff_all, dim=0)
    sol_all = torch.stack(sol_all, dim=0)

    x_train = coeff_all[:ntrain_eff]
    y_train = sol_all[:ntrain_eff]
    x_test = coeff_all[ntrain_eff:]
    y_test = sol_all[ntrain_eff:]
    return x_train, y_train, x_test, y_test


# ------------------------------------------------------------
# OF-RNO reservoir + readout
# ------------------------------------------------------------
class FixedOFRNO2d(nn.Module):
    def __init__(
        self,
        s: int,
        channels: int,
        num_layers: int,
        mask_modes: int,
        alpha: float,
        beta: float,
        nonlinearity: str,
    ) -> None:
        super().__init__()
        self.s = s
        self.channels = channels
        self.num_layers = num_layers
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.nonlinearity = nonlinearity

        scale = 1.0 / np.sqrt(max(1, channels))

        lift_w = scale * torch.randn(channels, 3)
        lift_b = 0.1 * torch.randn(channels)

        pre_mix = scale * torch.randn(num_layers, channels, channels)
        post_mix = scale * torch.randn(num_layers, channels, channels)
        layer_bias = 0.1 * torch.randn(num_layers, channels, 1, 1)

        lowfreq = _build_lowfreq_mask(s, mask_modes)

        phase = 2.0 * np.pi * torch.rand(num_layers, channels, s, s)
        amp = torch.rand(num_layers, channels, s, s)
        optical_masks = amp * torch.exp(1j * phase)
        optical_masks = optical_masks * lowfreq[None, None, :, :]

        self.register_buffer("lift_w", lift_w)
        self.register_buffer("lift_b", lift_b)
        self.register_buffer("pre_mix", pre_mix)
        self.register_buffer("post_mix", post_mix)
        self.register_buffer("layer_bias", layer_bias)
        self.register_buffer("lowfreq_mask", lowfreq)
        self.register_buffer("optical_masks", optical_masks.to(torch.complex64))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, s, s, 1) normalized coefficient
        b, sx, sy, c = x.shape
        if c != 1:
            raise ValueError(f"expected input channels=1, got {c}")
        if sx != self.s or sy != self.s:
            raise ValueError(f"grid size mismatch: expected {self.s}x{self.s}, got {sx}x{sy}")

        grid = _build_grid(b, sx, sy, x.device, x.dtype)
        inp = torch.cat((x, grid), dim=-1)  # (B, s, s, 3)

        v = torch.einsum("bxyi,ci->bcxy", inp, self.lift_w.to(x.dtype))
        v = v + self.lift_b.view(1, -1, 1, 1).to(x.dtype)
        v = _activation(v, self.nonlinearity)

        lowfreq_complex = self.lowfreq_mask.to(torch.complex64)
        for l in range(self.num_layers):
            v_mix = torch.einsum("ij,bjxy->bixy", self.pre_mix[l].to(v.dtype), v)
            v_ft = torch.fft.fft2(v_mix)
            filt = self.optical_masks[l]
            e = torch.fft.ifft2(v_ft * filt * lowfreq_complex)
            intensity = e.real.square() + e.imag.square()
            u = torch.einsum("ij,bjxy->bixy", self.post_mix[l].to(v.dtype), intensity)
            v = _activation(self.alpha * v + self.beta * u + self.layer_bias[l].to(v.dtype), self.nonlinearity)
        return v


class PointwiseReadout(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(channels))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        # v: (B, C, s, s) -> uhat: (B, s, s)
        return torch.einsum("bcxy,c->bxy", v, self.weight) + self.bias


# ------------------------------------------------------------
# Darcy operator and linear row construction
# ------------------------------------------------------------
def darcy_apply_La_batch(a: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """Vectorized Darcy operator L_a u on interior points.

    Args:
        a: (B, 1, s, s)
        u: (B, C, s, s)
    Returns:
        (B, C, s, s) with interior filled and boundary zero.
    """
    if a.dim() != 4 or u.dim() != 4:
        raise ValueError("a and u must be rank-4 tensors")
    if a.size(1) != 1:
        raise ValueError(f"a must have channel=1, got {a.size(1)}")

    b, c, s1, s2 = u.shape
    if a.shape[0] != b or a.shape[2] != s1 or a.shape[3] != s2:
        raise ValueError(f"shape mismatch: a={tuple(a.shape)} u={tuple(u.shape)}")

    h = 1.0 / (s1 - 1)
    inv_h2 = 1.0 / (h * h)

    aa = a[:, 0]
    uc = u[:, :, 1:-1, 1:-1]
    up = u[:, :, 2:, 1:-1]
    um = u[:, :, :-2, 1:-1]
    ur = u[:, :, 1:-1, 2:]
    ul = u[:, :, 1:-1, :-2]

    a_ip = 0.5 * (aa[:, 1:-1, 1:-1] + aa[:, 2:, 1:-1])
    a_im = 0.5 * (aa[:, 1:-1, 1:-1] + aa[:, :-2, 1:-1])
    a_jp = 0.5 * (aa[:, 1:-1, 1:-1] + aa[:, 1:-1, 2:])
    a_jm = 0.5 * (aa[:, 1:-1, 1:-1] + aa[:, 1:-1, :-2])

    interior = (
        a_ip.unsqueeze(1) * (uc - up)
        + a_im.unsqueeze(1) * (uc - um)
        + a_jp.unsqueeze(1) * (uc - ur)
        + a_jm.unsqueeze(1) * (uc - ul)
    ) * inv_h2

    out = torch.zeros_like(u)
    out[:, :, 1:-1, 1:-1] = interior
    return out


def _build_supervised_rows(v: torch.Tensor, y_phys: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # v: (B,C,s,s), y_phys: (B,s,s)
    b, c, s, _ = v.shape
    feat = v.permute(0, 2, 3, 1).reshape(b * s * s, c)
    ones = torch.ones(feat.size(0), 1, device=v.device, dtype=v.dtype)
    xmat = torch.cat((feat, ones), dim=1)
    yvec = y_phys.reshape(-1, 1)
    return xmat, yvec


def _build_pde_rows(
    v: torch.Tensor,
    a_phys: torch.Tensor,
    pde_samples: int,
    boundary_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    # v: (B,C,s,s), a_phys: (B,1,s,s), boundary_mask: (s,s) bool
    b, c, s, _ = v.shape
    la_v = darcy_apply_La_batch(a_phys, v)

    ones_field = torch.ones(b, 1, s, s, device=v.device, dtype=v.dtype)
    la_one = darcy_apply_La_batch(a_phys, ones_field)

    rows = []
    targets = []
    for i in range(b):
        interior_feat = la_v[i, :, 1:-1, 1:-1].reshape(c, -1).transpose(0, 1)
        interior_bias = la_one[i, 0, 1:-1, 1:-1].reshape(-1, 1)
        n_int = interior_feat.size(0)
        m = min(max(0, pde_samples), n_int)
        if m > 0:
            idx = torch.randperm(n_int, device=v.device)[:m]
            xi = torch.cat((interior_feat[idx], interior_bias[idx]), dim=1)
            yi = torch.ones(m, 1, device=v.device, dtype=v.dtype)
            rows.append(xi)
            targets.append(yi)

        # Boundary equations: u_hat = 0
        vb = v[i, :, boundary_mask].transpose(0, 1)
        xb = torch.cat((vb, torch.ones(vb.size(0), 1, device=v.device, dtype=v.dtype)), dim=1)
        yb = torch.zeros(vb.size(0), 1, device=v.device, dtype=v.dtype)
        rows.append(xb)
        targets.append(yb)

    if not rows:
        xmat = torch.empty(0, c + 1, device=v.device, dtype=v.dtype)
        yvec = torch.empty(0, 1, device=v.device, dtype=v.dtype)
        return xmat, yvec

    xmat = torch.cat(rows, dim=0)
    yvec = torch.cat(targets, dim=0)
    return xmat, yvec


# ------------------------------------------------------------
# Data pipeline
# ------------------------------------------------------------
def _load_data(args: argparse.Namespace):
    r = args.r
    h = int(((args.grid_size - 1) / r) + 1)
    s = h

    use_smoke = bool(args.smoke)
    missing = _file_missing(args)
    if missing:
        use_smoke = True

    if use_smoke:
        if args.smoke:
            print("[data] smoke mode enabled: using synthetic tiny dataset.")
        else:
            print("[data] dataset files missing: fallback to synthetic tiny dataset.")
        x_train_phys, y_train_phys, x_test_phys, y_test_phys = _generate_smoke_data(args.ntrain, args.ntest, s)
    else:
        if args.data_mode == "single_split":
            reader = MatReader(args.data_file)
            x_data = reader.read_field("coeff")[:, ::r, ::r][:, :s, :s]
            y_data = reader.read_field("sol")[:, ::r, ::r][:, :s, :s]

            x_train_phys = x_data[: args.ntrain]
            y_train_phys = y_data[: args.ntrain]
            x_test_phys = x_data[-args.ntest :]
            y_test_phys = y_data[-args.ntest :]
        else:
            reader = MatReader(args.train_file)
            x_train_phys = reader.read_field("coeff")[: args.ntrain, ::r, ::r][:, :s, :s]
            y_train_phys = reader.read_field("sol")[: args.ntrain, ::r, ::r][:, :s, :s]

            reader.load_file(args.test_file)
            x_test_phys = reader.read_field("coeff")[: args.ntest, ::r, ::r][:, :s, :s]
            y_test_phys = reader.read_field("sol")[: args.ntest, ::r, ::r][:, :s, :s]

    ntrain = x_train_phys.shape[0]
    ntest = x_test_phys.shape[0]
    s_eff = x_train_phys.shape[1]

    # Normalize x exactly like fourier_2d.py
    x_normalizer = UnitGaussianNormalizer(x_train_phys)
    x_train = x_normalizer.encode(x_train_phys)
    x_test = x_normalizer.encode(x_test_phys)

    # Keep y normalization pattern available for supervised SGD.
    y_normalizer = UnitGaussianNormalizer(y_train_phys)
    y_train_encoded = y_normalizer.encode(y_train_phys)

    x_train = x_train.reshape(ntrain, s_eff, s_eff, 1)
    x_test = x_test.reshape(ntest, s_eff, s_eff, 1)

    return {
        "s": s_eff,
        "ntrain": ntrain,
        "ntest": ntest,
        "use_smoke": use_smoke,
        "x_train": x_train.float(),
        "x_test": x_test.float(),
        "x_train_phys": x_train_phys.float(),
        "x_test_phys": x_test_phys.float(),
        "y_train_phys": y_train_phys.float(),
        "y_test_phys": y_test_phys.float(),
        "y_train_encoded": y_train_encoded.float(),
        "x_normalizer": x_normalizer,
        "y_normalizer": y_normalizer,
    }


# ------------------------------------------------------------
# Fit modes
# ------------------------------------------------------------
def _evaluate_rel_l2(
    reservoir: FixedOFRNO2d,
    readout: PointwiseReadout,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    y_index: int,
) -> float:
    myloss = LpLoss(size_average=False)
    total = 0.0
    count = 0

    reservoir.eval()
    readout.eval()
    with torch.no_grad():
        for batch in loader:
            x = batch[0].to(device)
            y = batch[y_index].to(device)
            v = reservoir(x)
            pred = readout(v)
            bsz = x.size(0)
            total += myloss(pred.reshape(bsz, -1), y.reshape(bsz, -1)).item()
            count += bsz

    if count == 0:
        return float("nan")
    return total / count


def fit_supervised_sgd(
    args: argparse.Namespace,
    reservoir: FixedOFRNO2d,
    readout: PointwiseReadout,
    train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    y_normalizer: UnitGaussianNormalizer,
    device: torch.device,
    ntrain: int,
) -> tuple[list[int], list[float], list[float]]:
    optimizer = torch.optim.Adam(readout.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    iterations = max(1, args.epochs * max(1, ntrain // max(1, args.batch_size)))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iterations)
    myloss = LpLoss(size_average=False)

    y_normalizer.to(device)

    hist_epochs: list[int] = []
    hist_train: list[float] = []
    hist_test: list[float] = []

    for ep in range(args.epochs):
        t1 = default_timer()
        train_l2 = 0.0

        reservoir.eval()
        readout.train()
        for x, y_enc, _y_phys, _a_phys in train_loader:
            x = x.to(device)
            y_enc = y_enc.to(device)
            y = y_normalizer.decode(y_enc)

            optimizer.zero_grad()
            with torch.no_grad():
                v = reservoir(x)
            pred = readout(v)
            bsz = x.size(0)
            loss = myloss(pred.reshape(bsz, -1), y.reshape(bsz, -1))
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_l2 += loss.item()

        train_l2 /= max(1, ntrain)
        test_l2 = _evaluate_rel_l2(reservoir, readout, test_loader, device, y_index=1)

        hist_epochs.append(ep)
        hist_train.append(train_l2)
        hist_test.append(test_l2)

        t2 = default_timer()
        print(ep, t2 - t1, train_l2, test_l2)

    return hist_epochs, hist_train, hist_test


def fit_supervised_ridge(
    args: argparse.Namespace,
    reservoir: FixedOFRNO2d,
    readout: PointwiseReadout,
    train_loader: torch.utils.data.DataLoader,
    y_normalizer: UnitGaussianNormalizer,
    device: torch.device,
) -> None:
    channels = reservoir.channels
    dim = channels + 1

    xtx = torch.zeros(dim, dim, device=device, dtype=torch.float64)
    xty = torch.zeros(dim, 1, device=device, dtype=torch.float64)

    y_normalizer.to(device)
    reservoir.eval()

    with torch.no_grad():
        for x, y_enc, _y_phys, _a_phys in train_loader:
            x = x.to(device)
            y_enc = y_enc.to(device)
            y_phys = y_normalizer.decode(y_enc)

            v = reservoir(x)
            xmat, yvec = _build_supervised_rows(v, y_phys)
            x64 = xmat.to(torch.float64)
            y64 = yvec.to(torch.float64)
            xtx += x64.transpose(0, 1) @ x64
            xty += x64.transpose(0, 1) @ y64

    reg = torch.eye(dim, device=device, dtype=torch.float64)
    reg[-1, -1] = 0.0
    theta = torch.linalg.solve(xtx + args.ridge_lam * reg, xty).squeeze(1)

    with torch.no_grad():
        readout.weight.copy_(theta[:-1].to(readout.weight.dtype))
        readout.bias.copy_(theta[-1:].to(readout.bias.dtype))


def fit_pde_ridge(
    args: argparse.Namespace,
    reservoir: FixedOFRNO2d,
    readout: PointwiseReadout,
    train_loader: torch.utils.data.DataLoader,
    device: torch.device,
    s: int,
) -> None:
    channels = reservoir.channels
    dim = channels + 1

    xtx = torch.zeros(dim, dim, device=device, dtype=torch.float64)
    xty = torch.zeros(dim, 1, device=device, dtype=torch.float64)

    bmask = _boundary_mask(s, device)
    reservoir.eval()

    with torch.no_grad():
        for x, _y_enc, _y_phys, a_phys in train_loader:
            x = x.to(device)
            a = a_phys.to(device).unsqueeze(1)
            v = reservoir(x)

            xmat, yvec = _build_pde_rows(v, a, args.pde_samples, bmask)
            if xmat.numel() == 0:
                continue
            x64 = xmat.to(torch.float64)
            y64 = yvec.to(torch.float64)
            xtx += x64.transpose(0, 1) @ x64
            xty += x64.transpose(0, 1) @ y64

    reg = torch.eye(dim, device=device, dtype=torch.float64)
    reg[-1, -1] = 0.0
    theta = torch.linalg.solve(xtx + args.ridge_lam * reg, xty).squeeze(1)

    with torch.no_grad():
        readout.weight.copy_(theta[:-1].to(readout.weight.dtype))
        readout.bias.copy_(theta[-1:].to(readout.bias.dtype))


def fit_pde_rls(
    args: argparse.Namespace,
    reservoir: FixedOFRNO2d,
    readout: PointwiseReadout,
    train_loader: torch.utils.data.DataLoader,
    device: torch.device,
    s: int,
) -> None:
    channels = reservoir.channels
    dim = channels + 1

    rho = float(args.rls_rho)
    delta = float(args.rls_delta)
    if not (0.0 < rho <= 1.0):
        raise ValueError("--rls-rho must be in (0, 1]")
    if delta <= 0.0:
        raise ValueError("--rls-delta must be > 0")

    theta = torch.zeros(dim, 1, device=device, dtype=torch.float64)
    pmat = (1.0 / delta) * torch.eye(dim, device=device, dtype=torch.float64)

    bmask = _boundary_mask(s, device)
    reservoir.eval()

    with torch.no_grad():
        for x, _y_enc, _y_phys, a_phys in train_loader:
            x = x.to(device)
            a = a_phys.to(device).unsqueeze(1)
            v = reservoir(x)

            xmat, yvec = _build_pde_rows(v, a, args.pde_samples, bmask)
            if xmat.numel() == 0:
                continue

            x64 = xmat.to(torch.float64)
            y64 = yvec.to(torch.float64)

            for i in range(x64.size(0)):
                z = x64[i : i + 1].transpose(0, 1)  # (dim,1)
                y = y64[i].item()
                denom = rho + (z.transpose(0, 1) @ pmat @ z).item()
                k = (pmat @ z) / denom
                err = y - (z.transpose(0, 1) @ theta).item()
                theta = theta + k * err
                pmat = (pmat - k @ z.transpose(0, 1) @ pmat) / rho

    with torch.no_grad():
        readout.weight.copy_(theta[:-1, 0].to(readout.weight.dtype))
        readout.bias.copy_(theta[-1:, 0].to(readout.bias.dtype))


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Digital PI-SC-OF-RNO for 2D Darcy")
    add_data_mode_args(
        parser,
        default_data_mode="separate_files",
        default_data_file="data/piececonst_r421_N1024_smooth1.mat",
        default_train_file="data/piececonst_r421_N1024_smooth1.mat",
        default_test_file="data/piececonst_r421_N1024_smooth2.mat",
    )

    parser.add_argument("--ntrain", type=int, default=1000, help="Number of training samples.")
    parser.add_argument("--ntest", type=int, default=100, help="Number of test samples.")
    parser.add_argument("--batch-size", type=int, default=20, help="Batch size.")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate for supervised_sgd.")
    parser.add_argument("--epochs", type=int, default=500, help="Epochs for supervised_sgd.")
    parser.add_argument("--width", type=int, default=32, help="Reservoir channel width C.")
    parser.add_argument("--r", type=int, default=5, help="Downsampling rate.")
    parser.add_argument("--grid-size", type=int, default=421, help="Original grid size (before downsampling).")

    parser.add_argument(
        "--fit-mode",
        type=str,
        default="supervised_sgd",
        choices=("supervised_sgd", "supervised_ridge", "pde_ridge", "pde_rls"),
        help="Readout fitting mode.",
    )
    parser.add_argument("--reservoir-layers", type=int, default=4, help="Number of fixed reservoir layers L.")
    parser.add_argument("--mask-modes", type=int, default=12, help="Low-frequency mask K for optical map.")
    parser.add_argument("--alpha", type=float, default=1.0, help="Residual scaling alpha.")
    parser.add_argument("--beta", type=float, default=1.0, help="Reservoir update scaling beta.")
    parser.add_argument(
        "--nonlinearity",
        type=str,
        default="tanh",
        choices=("tanh", "gelu", "relu", "identity"),
        help="Activation sigma in reservoir updates.",
    )

    parser.add_argument("--ridge-lam", type=float, default=1e-6, help="Ridge regularization lambda.")
    parser.add_argument("--pde-samples", type=int, default=2048, help="Interior collocation samples per sample for PDE modes.")
    parser.add_argument("--rls-rho", type=float, default=0.999, help="RLS forgetting factor rho.")
    parser.add_argument("--rls-delta", type=float, default=1.0, help="RLS initialization delta (P0=I/delta).")

    parser.add_argument("--smoke", action="store_true", help="Use synthetic tiny tensors for end-to-end smoke testing.")
    return parser


def _validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    validate_data_mode_args(args, parser)
    if args.r <= 0:
        parser.error("--r must be > 0")
    if args.grid_size <= 1:
        parser.error("--grid-size must be > 1")
    if args.width <= 0:
        parser.error("--width must be > 0")
    if args.reservoir_layers <= 0:
        parser.error("--reservoir-layers must be > 0")
    if args.mask_modes < 0:
        parser.error("--mask-modes must be >= 0")
    if args.pde_samples < 0:
        parser.error("--pde-samples must be >= 0")
    if args.ridge_lam < 0:
        parser.error("--ridge-lam must be >= 0")


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    _validate_args(args, parser)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    data = _load_data(args)
    s = data["s"]
    ntrain = data["ntrain"]
    ntest = data["ntest"]

    x_train = data["x_train"]
    x_test = data["x_test"]
    y_train_encoded = data["y_train_encoded"]
    y_train_phys = data["y_train_phys"]
    y_test_phys = data["y_test_phys"]
    a_train_phys = data["x_train_phys"]
    a_test_phys = data["x_test_phys"]

    x_normalizer = data["x_normalizer"]
    y_normalizer = data["y_normalizer"]

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_train, y_train_encoded, y_train_phys, a_train_phys),
        batch_size=args.batch_size,
        shuffle=True,
    )
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_test, y_test_phys, a_test_phys),
        batch_size=args.batch_size,
        shuffle=False,
    )

    reservoir = FixedOFRNO2d(
        s=s,
        channels=args.width,
        num_layers=args.reservoir_layers,
        mask_modes=args.mask_modes,
        alpha=args.alpha,
        beta=args.beta,
        nonlinearity=args.nonlinearity,
    ).to(device)

    # Freeze reservoir explicitly.
    for p in reservoir.parameters():
        p.requires_grad = False

    readout = PointwiseReadout(args.width).to(device)

    viz_dir = os.path.join("visualizations", "rno_2d")
    os.makedirs(viz_dir, exist_ok=True)

    hist_epochs: list[int] = []
    hist_train_rel_l2: list[float] = []
    hist_test_rel_l2: list[float] = []

    t_fit0 = default_timer()
    if args.fit_mode == "supervised_sgd":
        hist_epochs, hist_train_rel_l2, hist_test_rel_l2 = fit_supervised_sgd(
            args,
            reservoir,
            readout,
            train_loader,
            test_loader,
            y_normalizer,
            device,
            ntrain,
        )
    elif args.fit_mode == "supervised_ridge":
        fit_supervised_ridge(args, reservoir, readout, train_loader, y_normalizer, device)
    elif args.fit_mode == "pde_ridge":
        fit_pde_ridge(args, reservoir, readout, train_loader, device, s)
    elif args.fit_mode == "pde_rls":
        fit_pde_rls(args, reservoir, readout, train_loader, device, s)
    else:
        raise ValueError(f"unknown fit mode: {args.fit_mode}")
    t_fit1 = default_timer()
    print(f"[fit] mode={args.fit_mode} done in {t_fit1 - t_fit0:.2f}s")

    if args.fit_mode != "supervised_sgd":
        train_rel = _evaluate_rel_l2(reservoir, readout, train_loader, device, y_index=2)
        test_rel = _evaluate_rel_l2(reservoir, readout, test_loader, device, y_index=1)
        hist_epochs = [0]
        hist_train_rel_l2 = [train_rel]
        hist_test_rel_l2 = [test_rel]
        print(f"[eval] train_relL2={train_rel:.6f} test_relL2={test_rel:.6f}")

    # Learning curve (only if at least one point exists)
    if hist_epochs:
        try:
            plot_learning_curve(
                LearningCurve(
                    epochs=hist_epochs,
                    train=hist_train_rel_l2,
                    test=hist_test_rel_l2,
                    train_label="train (relL2)",
                    test_label="test (relL2)",
                    metric_name="relative L2",
                ),
                out_path_no_ext=os.path.join(viz_dir, "learning_curve_relL2"),
                logy=True,
                title=f"rno_2d ({args.fit_mode}): relative L2",
            )
        except Exception as e:
            print(f"[viz] learning curve failed: {e}")

    # Sample predictions + histogram
    try:
        reservoir.eval()
        readout.eval()

        sample_ids = [0, min(1, ntest - 1), min(2, ntest - 1)]
        per_sample_err: list[float] = []

        with torch.no_grad():
            for i in range(ntest):
                x_i = x_test[i : i + 1].to(device)
                y_i = y_test_phys[i : i + 1].to(device)

                pred_i = readout(reservoir(x_i))
                per_sample_err.append(rel_l2(pred_i.reshape(-1), y_i.reshape(-1)))

                if i in sample_ids:
                    coeff_i = x_normalizer.decode(x_test[i].squeeze(-1)).squeeze()
                    plot_2d_comparison(
                        gt=y_i.squeeze().cpu(),
                        pred=pred_i.squeeze().cpu(),
                        input_field=coeff_i.cpu(),
                        out_path_no_ext=os.path.join(viz_dir, f"sample_{i:03d}"),
                        suptitle=f"sample {i}  relL2={per_sample_err[-1]:.3g}",
                    )

        plot_error_histogram(per_sample_err, os.path.join(viz_dir, "test_relL2_hist"))
        print(f"[eval] test_relL2 mean={np.mean(per_sample_err):.6f}")
    except Exception as e:
        print(f"[viz] failed: {e}")


if __name__ == "__main__":
    main()

# Smoke test examples:
# python rno_2d.py --smoke --epochs 2 --fit-mode supervised_sgd
# python rno_2d.py --smoke --fit-mode pde_ridge --pde-samples 64
