import argparse

import numpy as np
import torch
from timeit import default_timer

from cli_utils import add_data_mode_args, validate_data_mode_args
from one.one_config import add_one_optical_args
from one.one_models import ONE2dDarcy


def _count_params(model: torch.nn.Module) -> int:
    total = 0
    for p in model.parameters():
        total += p.numel() * (2 if p.is_complex() else 1)
    return total


def _append_grid_channels(x: torch.Tensor) -> torch.Tensor:
    # third_party Darcy demo pre-concatenates grid before model forward.
    b, sx, sy = x.shape[0], x.shape[1], x.shape[2]
    gx = torch.linspace(0, 1, sx, dtype=x.dtype).view(1, sx, 1, 1).repeat(b, 1, sy, 1)
    gy = torch.linspace(0, 1, sy, dtype=x.dtype).view(1, 1, sy, 1).repeat(b, sx, 1, 1)
    return torch.cat((x, gx, gy), dim=-1)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ONE baseline for Darcy (2D)")
    add_data_mode_args(
        parser,
        default_data_mode="separate_files",
        default_data_file="data/piececonst_r421_N1024_smooth1.mat",
        default_train_file="data/piececonst_r421_N1024_smooth1.mat",
        default_test_file="data/piececonst_r421_N1024_smooth2.mat",
    )
    parser.add_argument("--ntrain", type=int, default=1000)
    parser.add_argument("--ntest", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--activation", choices=("tanh", "gelu"), default="tanh")
    parser.add_argument("--r", type=int, default=5)
    parser.add_argument("--grid-size", type=int, default=421)
    parser.add_argument("--smoke-test", action="store_true", help="Run one synthetic forward/backward pass and exit.")
    add_one_optical_args(parser)
    return parser


def _validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    validate_data_mode_args(args, parser)


def _run_smoke_test(args: argparse.Namespace, device: torch.device) -> None:
    s = 33
    in_channels = 3 if args.one_mode == "tp_compat" else 1
    model = ONE2dDarcy(
        spatial_size=s,
        width=8,
        in_channels=in_channels,
        domain_padding=args.domain_padding,
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

    x = torch.randn(2, s, s, 1, device=device)
    if args.one_mode == "tp_compat":
        x = _append_grid_channels(x)
    x = x.to(device)
    y = torch.randn(2, s, s, device=device)

    pred = model(x).reshape(x.shape[0], s, s)
    loss = torch.mean((pred - y) ** 2)
    opt.zero_grad()
    loss.backward()
    opt.step()
    print(f"[smoke-test] success: loss={loss.item():.6f}")


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    _validate_args(args, parser)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.smoke_test:
        _run_smoke_test(args, device)
        return

    from utilities3 import LpLoss, MatReader, UnitGaussianNormalizer

    torch.manual_seed(0)
    np.random.seed(0)

    ntrain = args.ntrain
    ntest = args.ntest

    batch_size = args.batch_size
    learning_rate = args.learning_rate
    epochs = args.epochs
    iterations = epochs * (ntrain // batch_size)

    r = args.r
    h = int(((args.grid_size - 1) / r) + 1)
    s = h

    if args.data_mode == "single_split":
        reader = MatReader(args.data_file)
        x_data = reader.read_field("coeff")[:, ::r, ::r][:, :s, :s]
        y_data = reader.read_field("sol")[:, ::r, ::r][:, :s, :s]

        x_train = x_data[:ntrain]
        y_train = y_data[:ntrain]
        x_test = x_data[-ntest:]
        y_test = y_data[-ntest:]
    else:
        reader = MatReader(args.train_file)
        x_train = reader.read_field("coeff")[:ntrain, ::r, ::r][:, :s, :s]
        y_train = reader.read_field("sol")[:ntrain, ::r, ::r][:, :s, :s]

        reader.load_file(args.test_file)
        x_test = reader.read_field("coeff")[:ntest, ::r, ::r][:, :s, :s]
        y_test = reader.read_field("sol")[:ntest, ::r, ::r][:, :s, :s]

    x_normalizer = UnitGaussianNormalizer(x_train)
    x_train = x_normalizer.encode(x_train)
    x_test = x_normalizer.encode(x_test)

    y_normalizer = UnitGaussianNormalizer(y_train)
    y_train = y_normalizer.encode(y_train)

    x_train = x_train.reshape(ntrain, s, s, 1)
    x_test = x_test.reshape(ntest, s, s, 1)
    if args.one_mode == "tp_compat":
        x_train = _append_grid_channels(x_train)
        x_test = _append_grid_channels(x_test)

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False
    )

    model = ONE2dDarcy(
        spatial_size=s,
        width=args.width,
        in_channels=(3 if args.one_mode == "tp_compat" else 1),
        domain_padding=args.domain_padding,
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
    y_normalizer.to(device)

    for ep in range(epochs):
        model.train()
        t1 = default_timer()
        train_l2 = 0.0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            bs = x.shape[0]

            optimizer.zero_grad()
            out = model(x).reshape(bs, s, s)
            out = y_normalizer.decode(out)
            y_dec = y_normalizer.decode(y)

            loss = myloss(out.reshape(bs, -1), y_dec.reshape(bs, -1))
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_l2 += loss.item()

        model.eval()
        test_l2 = 0.0
        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(device)
                y = y.to(device)
                bs = x.shape[0]

                out = model(x).reshape(bs, s, s)
                out = y_normalizer.decode(out)
                test_l2 += myloss(out.reshape(bs, -1), y.reshape(bs, -1)).item()

        train_l2 /= ntrain
        test_l2 /= ntest
        t2 = default_timer()
        print(ep, t2 - t1, train_l2, test_l2)


if __name__ == "__main__":
    main()
