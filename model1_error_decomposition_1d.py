from __future__ import annotations

import argparse

from pol.model123_1d.error_decomposition import ErrorDecompositionConfig, run_error_decomposition


def _parse_csv_values(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _build_ttilde_range(start: float, stop: float, step: float) -> list[float]:
    if step <= 0.0:
        raise ValueError("ttilde-step must be positive")
    if stop < start:
        raise ValueError("ttilde-stop must be >= ttilde-start")
    values: list[float] = []
    count = int(round((stop - start) / step))
    for idx in range(count + 1):
        values.append(round(start + idx * step, 12))
    if values[-1] != round(stop, 12):
        values.append(round(stop, 12))
    return values


def _parse_ttilde_values(args: argparse.Namespace) -> list[float]:
    if args.ttilde_values.strip():
        values = [float(item) for item in _parse_csv_values(args.ttilde_values)]
    elif args.ttilde_start is not None and args.ttilde_stop is not None:
        values = _build_ttilde_range(args.ttilde_start, args.ttilde_stop, args.ttilde_step)
    else:
        values = [float(args.T)]
    if not values:
        raise ValueError("No Ttilde values were provided")
    for value in values:
        if value <= 0.0:
            raise ValueError("All Ttilde values must be positive")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Model 1 error decomposition experiment for 1D periodic Burgers"
    )
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--nx", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--target-nu", type=float, default=0.05)
    parser.add_argument("--T", type=float, default=1.0)
    parser.add_argument("--ttilde-values", type=str, default="")
    parser.add_argument("--ttilde-start", type=float, default=None)
    parser.add_argument("--ttilde-stop", type=float, default=None)
    parser.add_argument("--ttilde-step", type=float, default=0.05)
    parser.add_argument("--dt", type=float, default=1e-2)
    parser.add_argument("--fine-dt", type=float, default=1e-3)
    parser.add_argument("--reservoir", choices=("burgers", "reaction_diffusion", "ks"), default="burgers")
    parser.add_argument("--rd-nu", type=float, default=1e-3)
    parser.add_argument("--rd-alpha", type=float, default=1.0)
    parser.add_argument("--rd-beta", type=float, default=1.0)
    parser.add_argument("--res-burgers-nu", type=float, default=0.05)
    parser.add_argument("--res-burgers-b", type=float, default=1.0)
    parser.add_argument("--ks-b", type=float, default=1.0)
    parser.add_argument("--ks-eta", type=float, default=1.0)
    parser.add_argument("--ks-kappa", type=float, default=1.0)
    parser.add_argument("--ks-dealias", action="store_true")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    parser.add_argument("--initial-condition-type", choices=("fourier", "grf"), default="fourier")
    parser.add_argument("--grf-gamma", type=float, default=2.0)
    parser.add_argument("--grf-tau", type=float, default=5.0)
    parser.add_argument("--grf-sigma", type=float, default=25.0)
    parser.add_argument("--grf-mean", type=float, default=0.0)
    parser.add_argument("--beta-mode", choices=("correlation", "empirical", "both"), default="both")
    parser.add_argument("--beta-max-states", type=int, default=24)
    parser.add_argument("--out-dir", default="outputs/model1_error_decomposition_1d")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    cfg = ErrorDecompositionConfig(
        num_samples=args.num_samples,
        nx=args.nx,
        seed=args.seed,
        batch_size=args.batch_size,
        target_nu=args.target_nu,
        T=args.T,
        Ttilde_values=_parse_ttilde_values(args),
        dt=args.dt,
        fine_dt=args.fine_dt,
        reservoir=args.reservoir,
        rd_nu=args.rd_nu,
        rd_alpha=args.rd_alpha,
        rd_beta=args.rd_beta,
        res_burgers_nu=args.res_burgers_nu,
        res_burgers_b=args.res_burgers_b,
        ks_b=args.ks_b,
        ks_eta=args.ks_eta,
        ks_kappa=args.ks_kappa,
        ks_dealias=args.ks_dealias,
        dtype=args.dtype,
        device=args.device,
        initial_condition_type=args.initial_condition_type,
        grf_gamma=args.grf_gamma,
        grf_tau=args.grf_tau,
        grf_sigma=args.grf_sigma,
        grf_mean=args.grf_mean,
        beta_mode=args.beta_mode,
        beta_max_states=args.beta_max_states,
        out_dir=args.out_dir,
    )
    result = run_error_decomposition(cfg, save_outputs=True)
    for row in result["summary_rows"]:
        print(
            "Ttilde=%g D1=%.6e matched=%.6e Delta_time=%.6e Delta_dyn=%.6e beta_emp=%.6e"
            % (
                row["Ttilde"],
                row["D1"],
                row["matched_time_error"],
                row["Delta_time"],
                row["Delta_dyn"],
                row["beta_empirical"],
            ),
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
