import argparse
import os
from timeit import default_timer

import numpy as np
import torch

from cli_utils import add_data_mode_args, add_split_args, validate_data_mode_args
from residual_models_1d import ELMResidual1D, ResidualCNN1D, ResidualFNO1D, ResidualMLP1D
from subordination_1d_time import BernsteinPsi, SubordinatedHeatOperator1D
from utilities3 import MatReader, count_params
from viz_utils import LearningCurve, plot_1d_prediction_multi, plot_error_histogram, plot_learning_curve, rel_l2


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


def _read_time_or_default(
    reader: MatReader,
    u_data: torch.Tensor,
    default_t_max: float,
    source_name: str,
) -> torch.Tensor:
    """Read 1D time array t; if missing, create uniform time grid from u's time length."""
    if u_data.ndim != 3:
        raise ValueError(f"Expected u=(N,S,T) before reading time array, got {tuple(u_data.shape)}")
    T = int(u_data.shape[-1])
    if T <= 1:
        raise ValueError(f"Expected time dimension T>1 in u, got T={T}")

    try:
        t = reader.read_field("t").reshape(-1)
    except KeyError:
        t = torch.linspace(0.0, float(default_t_max), T, dtype=u_data.dtype)
        print(
            f"[warn] 't' field not found in {source_name}. "
            f"Using fallback t=linspace(0,{float(default_t_max):g},{T})."
        )
        return t

    if t.ndim != 1:
        raise ValueError(f"Expected t=(T,), got {tuple(t.shape)} in {source_name}")
    if t.numel() != T:
        raise ValueError(
            f"Time length mismatch in {source_name}: u has T={T} but t has length {t.numel()}."
        )
    return t


def _load_single_split(
    args: argparse.Namespace,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float | None]:
    reader = MatReader(args.data_file)
    a_data = reader.read_field("a")
    u_data = reader.read_field("u")
    t = _read_time_or_default(reader, u_data, args.default_t_max, args.data_file)
    alpha = _read_optional_scalar(reader, "alpha")

    if a_data.ndim != 2 or u_data.ndim != 3:
        raise ValueError(f"Expected a=(N,S), u=(N,S,T), got {tuple(a_data.shape)}, {tuple(u_data.shape)}")
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
    t_train = _read_time_or_default(train_reader, train_reader.read_field("u"), args.default_t_max, args.train_file)[
        :: args.sub_t
    ]

    a_test_all = test_reader.read_field("a")[:, :: args.sub]
    u_test_all = test_reader.read_field("u")[:, :: args.sub, :: args.sub_t]
    t_test = _read_time_or_default(test_reader, test_reader.read_field("u"), args.default_t_max, args.test_file)[
        :: args.sub_t
    ]

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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="1D hybrid subordination + residual correction")
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
    parser.add_argument(
        "--default-t-max",
        type=float,
        default=1.0,
        help="Fallback max time used only when 't' field is missing in .mat (uniform linspace).",
    )

    parser.add_argument("--psi-J", type=int, default=32)
    parser.add_argument("--learn-s", action="store_true")
    parser.add_argument("--psi-s-min", type=float, default=1e-3)
    parser.add_argument("--psi-s-max", type=float, default=1e3)
    parser.add_argument("--psi-eps", type=float, default=1e-8)

    parser.add_argument("--base-epochs", type=int, default=200)
    parser.add_argument("--base-lr", type=float, default=1e-2)
    parser.add_argument("--base-batch-size", type=int, default=20)

    parser.add_argument("--residual", choices=("none", "mlp", "cnn", "fno", "elm"), default="none")
    parser.add_argument("--res-input", choices=("a_only", "base_only", "a_and_base"), default="a_and_base")
    parser.add_argument("--train-mode", choices=("two_stage", "joint"), default="two_stage")

    parser.add_argument("--res-epochs", type=int, default=200)
    parser.add_argument("--res-lr", type=float, default=1e-3)
    parser.add_argument("--res-batch-size", type=int, default=20)
    parser.add_argument("--freeze-psi", action="store_true")

    parser.add_argument("--mlp-width", type=int, default=256)
    parser.add_argument("--mlp-depth", type=int, default=4)
    parser.add_argument("--mlp-include-x", action="store_true")

    parser.add_argument("--cnn-width", type=int, default=128)
    parser.add_argument("--cnn-depth", type=int, default=4)
    parser.add_argument("--cnn-kernel", type=int, default=5)
    parser.add_argument("--cnn-include-x", action="store_true")

    parser.add_argument("--fno-width", type=int, default=64)
    parser.add_argument("--fno-modes", type=int, default=16)
    parser.add_argument("--fno-layers", type=int, default=4)

    parser.add_argument("--elm-hidden", type=int, default=2000)
    parser.add_argument("--elm-lam", type=float, default=1e-6)
    parser.add_argument("--elm-act", choices=("tanh", "relu", "gelu"), default="tanh")
    parser.add_argument("--elm-seed", type=int, default=0)
    parser.add_argument("--elm-standardize-x", action="store_true")

    parser.add_argument("--viz-dir", type=str, default="visualizations/hybrid_subordination_1d_time")
    parser.add_argument("--plot-samples", type=int, default=3)
    parser.add_argument("--plot-times", type=str, default="")
    return parser


def _validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    validate_data_mode_args(args, parser)
    if args.sub <= 0:
        parser.error("--sub must be positive.")
    if args.sub_t <= 0:
        parser.error("--sub-t must be positive.")
    if args.ntrain <= 0 or args.ntest <= 0:
        parser.error("--ntrain and --ntest must be positive.")
    if args.default_t_max <= 0:
        parser.error("--default-t-max must be positive.")

    if args.base_epochs <= 0:
        parser.error("--base-epochs must be positive.")
    if args.base_lr <= 0:
        parser.error("--base-lr must be positive.")
    if args.base_batch_size <= 0:
        parser.error("--base-batch-size must be positive.")

    if args.residual in {"mlp", "cnn", "fno"}:
        if args.res_epochs <= 0:
            parser.error("--res-epochs must be positive for nn residuals.")
        if args.res_lr <= 0:
            parser.error("--res-lr must be positive for nn residuals.")
        if args.res_batch_size <= 0:
            parser.error("--res-batch-size must be positive for nn residuals.")

    if args.train_mode == "joint":
        if args.residual in {"none", "elm"}:
            parser.error("--train-mode joint is only supported for residual in {mlp,cnn,fno}.")


def _set_requires_grad(module: torch.nn.Module, flag: bool) -> None:
    for p in module.parameters():
        p.requires_grad = flag


def _parse_plot_times(plot_times_str: str, T: int) -> list[int]:
    if not plot_times_str:
        return sorted(set([0, T // 2, T - 1]))
    idxs = []
    for token in plot_times_str.split(","):
        token = token.strip()
        if not token:
            continue
        idx = int(token)
        if idx < 0 or idx >= T:
            raise ValueError(f"Invalid plot time index {idx}. Valid range is [0, {T - 1}].")
        idxs.append(idx)
    if not idxs:
        raise ValueError("--plot-times was provided but no valid indices were parsed.")
    return sorted(set(idxs))


def _build_res_input(a: torch.Tensor, u_base: torch.Tensor, mode: str) -> torch.Tensor:
    # a: (B,S,1), u_base: (B,S,T) -> I: (B,S,C)
    if mode == "a_only":
        return a
    if mode == "base_only":
        return u_base
    if mode == "a_and_base":
        return torch.cat([a, u_base], dim=-1)
    raise ValueError(f"Unknown res-input mode: {mode}")


def _res_in_channels(mode: str, T: int) -> int:
    if mode == "a_only":
        return 1
    if mode == "base_only":
        return T
    if mode == "a_and_base":
        return T + 1
    raise ValueError(f"Unknown res-input mode: {mode}")


def _build_elm_features(a: torch.Tensor, u_base: torch.Tensor, mode: str) -> torch.Tensor:
    # a: (N,S,1), u_base: (N,S,T)
    a_vec = a[..., 0].reshape(a.shape[0], -1)
    ub_vec = u_base.reshape(u_base.shape[0], -1)
    if mode == "a_only":
        return a_vec
    if mode == "base_only":
        return ub_vec
    if mode == "a_and_base":
        return torch.cat([a_vec, ub_vec], dim=1)
    raise ValueError(f"Unknown res-input mode: {mode}")


def _build_residual_model(args: argparse.Namespace, in_channels: int, out_channels: int) -> torch.nn.Module:
    if args.residual == "mlp":
        return ResidualMLP1D(
            in_channels=in_channels,
            out_channels=out_channels,
            width=args.mlp_width,
            depth=args.mlp_depth,
            include_x=args.mlp_include_x,
        )
    if args.residual == "cnn":
        return ResidualCNN1D(
            in_channels=in_channels,
            out_channels=out_channels,
            width=args.cnn_width,
            depth=args.cnn_depth,
            kernel_size=args.cnn_kernel,
            include_x=args.cnn_include_x,
        )
    if args.residual == "fno":
        return ResidualFNO1D(
            in_channels=in_channels,
            out_channels=out_channels,
            width=args.fno_width,
            modes=args.fno_modes,
            n_layers=args.fno_layers,
        )
    raise ValueError(f"Residual model {args.residual} is not a torch.nn.Module residual.")


def _predict_base(model: SubordinatedHeatOperator1D, a: torch.Tensor, t_dev: torch.Tensor, batch_size: int, device: torch.device) -> torch.Tensor:
    loader = torch.utils.data.DataLoader(a, batch_size=batch_size, shuffle=False)
    out: list[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for a_batch in loader:
            a_batch = a_batch.to(device)
            out.append(model(a_batch, t_dev).cpu())
    return torch.cat(out, dim=0)


def _predict_hybrid_nn(
    base_model: SubordinatedHeatOperator1D,
    residual_model: torch.nn.Module,
    a: torch.Tensor,
    t_dev: torch.Tensor,
    mode: str,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    loader = torch.utils.data.DataLoader(a, batch_size=batch_size, shuffle=False)
    base_out: list[torch.Tensor] = []
    hyb_out: list[torch.Tensor] = []
    base_model.eval()
    residual_model.eval()

    with torch.no_grad():
        for a_batch in loader:
            a_batch = a_batch.to(device)
            u_base = base_model(a_batch, t_dev)
            inp = _build_res_input(a_batch, u_base, mode)
            res = residual_model(inp)
            base_out.append(u_base.cpu())
            hyb_out.append((u_base + res).cpu())

    return torch.cat(base_out, dim=0), torch.cat(hyb_out, dim=0)


def _dataset_rel_l2(pred: torch.Tensor, gt: torch.Tensor) -> float:
    return float(_relative_l2_batch(pred, gt).item())


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    _validate_args(args, parser)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.data_mode == "single_split":
        x_train, y_train, x_test, y_test, t, _ = _load_single_split(args)
    else:
        x_train, y_train, x_test, y_test, t, _ = _load_separate_files(args)

    # Shapes:
    # x_*: (N,S) -> (N,S,1), y_*: (N,S,T)
    x_train = x_train.unsqueeze(-1)
    x_test = x_test.unsqueeze(-1)

    S = int(x_train.shape[1])
    T = int(y_train.shape[-1])
    if y_train.shape[1] != S or y_test.shape[1] != S:
        raise ValueError("Spatial shape mismatch between input a and target u.")
    if t.numel() != T:
        raise ValueError(f"Time dimension mismatch: y has T={T}, t has {t.numel()}.")

    plot_t_indices = _parse_plot_times(args.plot_times, T)
    os.makedirs(args.viz_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t_dev = t.to(device)

    psi = BernsteinPsi(
        J=args.psi_J,
        s_min=args.psi_s_min,
        s_max=args.psi_s_max,
        learn_s=args.learn_s,
        s_eps=args.psi_eps,
    )
    base_model = SubordinatedHeatOperator1D(S=S, psi=psi).to(device)

    print(f"device={device}, base_params={count_params(base_model)}")
    print(f"residual={args.residual}, train_mode={args.train_mode}, res_input={args.res_input}")

    base_train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_train, y_train), batch_size=args.base_batch_size, shuffle=True
    )
    base_test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_test, y_test), batch_size=args.base_batch_size, shuffle=False
    )

    # Stage-1: base training (psi only)
    opt_base = torch.optim.Adam(base_model.parameters(), lr=args.base_lr)
    base_epochs_hist: list[int] = []
    base_train_hist: list[float] = []
    base_test_hist: list[float] = []

    for ep in range(args.base_epochs):
        t0 = default_timer()
        base_model.train()
        train_sum = 0.0
        train_count = 0
        for a_batch, u_batch in base_train_loader:
            a_batch = a_batch.to(device)
            u_batch = u_batch.to(device)

            opt_base.zero_grad()
            pred_base = base_model(a_batch, t_dev)
            loss = _relative_l2_batch(pred_base, u_batch)
            loss.backward()
            opt_base.step()

            bs = a_batch.shape[0]
            train_sum += float(loss.item()) * bs
            train_count += bs

        base_model.eval()
        test_sum = 0.0
        test_count = 0
        with torch.no_grad():
            for a_batch, u_batch in base_test_loader:
                a_batch = a_batch.to(device)
                u_batch = u_batch.to(device)
                pred_base = base_model(a_batch, t_dev)
                loss = _relative_l2_batch(pred_base, u_batch)
                bs = a_batch.shape[0]
                test_sum += float(loss.item()) * bs
                test_count += bs

        train_rel = train_sum / max(1, train_count)
        test_rel = test_sum / max(1, test_count)
        base_epochs_hist.append(ep)
        base_train_hist.append(train_rel)
        base_test_hist.append(test_rel)

        t1 = default_timer()
        print(f"[base] ep={ep:04d} time={t1 - t0:.2f}s train_relL2={train_rel:.6f} test_relL2={test_rel:.6f}")

    plot_learning_curve(
        LearningCurve(
            epochs=base_epochs_hist,
            train=base_train_hist,
            test=base_test_hist,
            train_label="base train (relL2)",
            test_label="base test (relL2)",
            metric_name="relative L2",
        ),
        out_path_no_ext=os.path.join(args.viz_dir, "learning_curve_base_relL2"),
        logy=True,
        title="hybrid_subordination_1d_time: base",
    )

    # Residual stage
    hybrid_epochs_hist: list[int] = []
    hybrid_train_hist: list[float] = []
    hybrid_test_hist: list[float] = []

    residual_model = None
    elm_model = None

    if args.residual == "none":
        pass

    elif args.residual == "elm":
        if args.train_mode != "two_stage":
            raise ValueError("ELM residual supports only --train-mode two_stage.")

        pred_base_train = _predict_base(base_model, x_train, t_dev, args.base_batch_size, device)
        pred_base_test = _predict_base(base_model, x_test, t_dev, args.base_batch_size, device)

        X_train = _build_elm_features(x_train, pred_base_train, args.res_input)
        Y_train = (y_train - pred_base_train).reshape(y_train.shape[0], -1)

        X_test = _build_elm_features(x_test, pred_base_test, args.res_input)

        elm_model = ELMResidual1D(
            input_dim=X_train.shape[1],
            output_dim=Y_train.shape[1],
            hidden_dim=args.elm_hidden,
            lam=args.elm_lam,
            activation=args.elm_act,
            seed=args.elm_seed,
            standardize_x=args.elm_standardize_x,
        )
        elm_model.fit(X_train, Y_train)

        r_train = elm_model.predict(X_train).reshape(y_train.shape[0], S, T)
        r_test = elm_model.predict(X_test).reshape(y_test.shape[0], S, T)
        pred_hybrid_train = pred_base_train + r_train
        pred_hybrid_test = pred_base_test + r_test

        hybrid_epochs_hist = [0]
        hybrid_train_hist = [_dataset_rel_l2(pred_hybrid_train, y_train)]
        hybrid_test_hist = [_dataset_rel_l2(pred_hybrid_test, y_test)]

        print(
            "[elm] "
            f"hidden={args.elm_hidden} lam={args.elm_lam} act={args.elm_act} "
            f"standardize_x={args.elm_standardize_x}"
        )

    else:
        in_channels = _res_in_channels(args.res_input, T)
        residual_model = _build_residual_model(args, in_channels=in_channels, out_channels=T).to(device)
        print(f"residual_params={count_params(residual_model)}")

        res_train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(x_train, y_train), batch_size=args.res_batch_size, shuffle=True
        )
        res_test_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(x_test, y_test), batch_size=args.res_batch_size, shuffle=False
        )

        if args.train_mode == "two_stage":
            _set_requires_grad(base_model.psi, False)
            _set_requires_grad(base_model, False)
            _set_requires_grad(residual_model, True)
            opt_res = torch.optim.Adam(residual_model.parameters(), lr=args.res_lr)

            for ep in range(args.res_epochs):
                t0 = default_timer()
                residual_model.train()
                train_sum = 0.0
                train_count = 0

                for a_batch, u_batch in res_train_loader:
                    a_batch = a_batch.to(device)
                    u_batch = u_batch.to(device)

                    with torch.no_grad():
                        u_base = base_model(a_batch, t_dev)
                    inp = _build_res_input(a_batch, u_base, args.res_input)

                    opt_res.zero_grad()
                    r = residual_model(inp)
                    u_pred = u_base + r
                    loss = _relative_l2_batch(u_pred, u_batch)
                    loss.backward()
                    opt_res.step()

                    bs = a_batch.shape[0]
                    train_sum += float(loss.item()) * bs
                    train_count += bs

                residual_model.eval()
                test_sum = 0.0
                test_count = 0
                with torch.no_grad():
                    for a_batch, u_batch in res_test_loader:
                        a_batch = a_batch.to(device)
                        u_batch = u_batch.to(device)

                        u_base = base_model(a_batch, t_dev)
                        inp = _build_res_input(a_batch, u_base, args.res_input)
                        r = residual_model(inp)
                        u_pred = u_base + r
                        loss = _relative_l2_batch(u_pred, u_batch)

                        bs = a_batch.shape[0]
                        test_sum += float(loss.item()) * bs
                        test_count += bs

                train_rel = train_sum / max(1, train_count)
                test_rel = test_sum / max(1, test_count)

                hybrid_epochs_hist.append(ep)
                hybrid_train_hist.append(train_rel)
                hybrid_test_hist.append(test_rel)

                t1 = default_timer()
                print(
                    f"[res-two-stage:{args.residual}] ep={ep:04d} time={t1 - t0:.2f}s "
                    f"train_relL2={train_rel:.6f} test_relL2={test_rel:.6f}"
                )

        else:
            _set_requires_grad(base_model, True)
            _set_requires_grad(residual_model, True)
            opt_joint = torch.optim.Adam(
                [
                    {"params": base_model.parameters(), "lr": args.base_lr},
                    {"params": residual_model.parameters(), "lr": args.res_lr},
                ]
            )

            for ep in range(args.res_epochs):
                t0 = default_timer()
                base_model.train()
                residual_model.train()
                train_sum = 0.0
                train_count = 0

                for a_batch, u_batch in res_train_loader:
                    a_batch = a_batch.to(device)
                    u_batch = u_batch.to(device)

                    opt_joint.zero_grad()
                    u_base = base_model(a_batch, t_dev)
                    inp = _build_res_input(a_batch, u_base, args.res_input)
                    r = residual_model(inp)
                    u_pred = u_base + r
                    loss = _relative_l2_batch(u_pred, u_batch)
                    loss.backward()
                    opt_joint.step()

                    bs = a_batch.shape[0]
                    train_sum += float(loss.item()) * bs
                    train_count += bs

                base_model.eval()
                residual_model.eval()
                test_sum = 0.0
                test_count = 0
                with torch.no_grad():
                    for a_batch, u_batch in res_test_loader:
                        a_batch = a_batch.to(device)
                        u_batch = u_batch.to(device)

                        u_base = base_model(a_batch, t_dev)
                        inp = _build_res_input(a_batch, u_base, args.res_input)
                        r = residual_model(inp)
                        u_pred = u_base + r
                        loss = _relative_l2_batch(u_pred, u_batch)

                        bs = a_batch.shape[0]
                        test_sum += float(loss.item()) * bs
                        test_count += bs

                train_rel = train_sum / max(1, train_count)
                test_rel = test_sum / max(1, test_count)

                hybrid_epochs_hist.append(ep)
                hybrid_train_hist.append(train_rel)
                hybrid_test_hist.append(test_rel)

                t1 = default_timer()
                print(
                    f"[res-joint:{args.residual}] ep={ep:04d} time={t1 - t0:.2f}s "
                    f"train_relL2={train_rel:.6f} test_relL2={test_rel:.6f}"
                )

    # Final predictions
    pred_base_train = _predict_base(base_model, x_train, t_dev, args.base_batch_size, device)
    pred_base_test = _predict_base(base_model, x_test, t_dev, args.base_batch_size, device)

    if args.residual == "none":
        pred_hybrid_train = pred_base_train
        pred_hybrid_test = pred_base_test
        hybrid_epochs_hist = [0]
        hybrid_train_hist = [_dataset_rel_l2(pred_hybrid_train, y_train)]
        hybrid_test_hist = [_dataset_rel_l2(pred_hybrid_test, y_test)]

    elif args.residual == "elm":
        # Already computed in ELM block.
        pass

    else:
        if residual_model is None:
            raise RuntimeError("Internal error: residual_model is None for nn residual type.")
        _, pred_hybrid_train = _predict_hybrid_nn(
            base_model=base_model,
            residual_model=residual_model,
            a=x_train,
            t_dev=t_dev,
            mode=args.res_input,
            batch_size=args.res_batch_size,
            device=device,
        )
        _, pred_hybrid_test = _predict_hybrid_nn(
            base_model=base_model,
            residual_model=residual_model,
            a=x_test,
            t_dev=t_dev,
            mode=args.res_input,
            batch_size=args.res_batch_size,
            device=device,
        )

    plot_learning_curve(
        LearningCurve(
            epochs=hybrid_epochs_hist,
            train=hybrid_train_hist,
            test=hybrid_test_hist,
            train_label="hybrid train (relL2)",
            test_label="hybrid test (relL2)",
            metric_name="relative L2",
        ),
        out_path_no_ext=os.path.join(args.viz_dir, "learning_curve_hybrid_relL2"),
        logy=True,
        title="hybrid_subordination_1d_time: hybrid",
    )

    base_train_rel = _dataset_rel_l2(pred_base_train, y_train)
    base_test_rel = _dataset_rel_l2(pred_base_test, y_test)
    hybrid_train_rel = _dataset_rel_l2(pred_hybrid_train, y_train)
    hybrid_test_rel = _dataset_rel_l2(pred_hybrid_test, y_test)

    print(f"base train relL2:   {base_train_rel:.6f}")
    print(f"base test relL2:    {base_test_rel:.6f}")
    print(f"hybrid train relL2: {hybrid_train_rel:.6f}")
    print(f"hybrid test relL2:  {hybrid_test_rel:.6f}")

    if args.residual == "mlp":
        print(f"residual hyperparams: width={args.mlp_width}, depth={args.mlp_depth}, include_x={args.mlp_include_x}")
    elif args.residual == "cnn":
        print(
            f"residual hyperparams: width={args.cnn_width}, depth={args.cnn_depth}, "
            f"kernel={args.cnn_kernel}, include_x={args.cnn_include_x}"
        )
    elif args.residual == "fno":
        print(f"residual hyperparams: width={args.fno_width}, modes={args.fno_modes}, layers={args.fno_layers}")
    elif args.residual == "elm":
        print(
            f"residual hyperparams: hidden={args.elm_hidden}, lam={args.elm_lam}, "
            f"act={args.elm_act}, seed={args.elm_seed}, standardize_x={args.elm_standardize_x}"
        )

    per_sample_base_err = [rel_l2(pred_base_test[i], y_test[i]) for i in range(pred_base_test.shape[0])]
    per_sample_hybrid_err = [rel_l2(pred_hybrid_test[i], y_test[i]) for i in range(pred_hybrid_test.shape[0])]

    plot_error_histogram(
        per_sample_base_err,
        os.path.join(args.viz_dir, "error_hist_base_test_relL2"),
        title="base test relL2 histogram",
    )
    plot_error_histogram(
        per_sample_hybrid_err,
        os.path.join(args.viz_dir, "error_hist_hybrid_test_relL2"),
        title="hybrid test relL2 histogram",
    )

    sample_count = max(1, min(args.plot_samples, x_test.shape[0]))
    x_grid = np.linspace(0.0, 1.0, S, endpoint=False)
    for sid in range(sample_count):
        for tidx in plot_t_indices:
            plot_1d_prediction_multi(
                x=x_grid,
                gt=y_test[sid, :, tidx],
                preds={
                    "base": pred_base_test[sid, :, tidx],
                    "hybrid": pred_hybrid_test[sid, :, tidx],
                },
                input_u0=x_test[sid, :, 0],
                out_path_no_ext=os.path.join(args.viz_dir, f"sample_{sid}_t{tidx}_hybrid"),
                title_prefix=f"sample={sid}, t_idx={tidx}:",
            )

    print(f"Saved visualizations to: {args.viz_dir}")


if __name__ == "__main__":
    main()
