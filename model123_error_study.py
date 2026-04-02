from __future__ import annotations

import argparse

from pol.model123_1d import ExperimentConfig, run_experiment


def parse_args() -> ExperimentConfig:
    parser = argparse.ArgumentParser(description="Model 1-3 runner for 1D Burgers target")
    parser.add_argument("--total-samples", type=int, default=1200)
    parser.add_argument("--ntrain", type=int, default=1000)
    parser.add_argument("--ntest", type=int, default=200)
    parser.add_argument("--nx", type=int, default=256)
    parser.add_argument("--dt", type=float, default=1e-3)
    parser.add_argument("--fine-dt", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--obs", choices=("full", "fourier"), default="fourier")
    parser.add_argument("--J", type=int, default=33)
    parser.add_argument("--K", type=int, default=1)
    parser.add_argument("--feature-times", type=str, default="")
    parser.add_argument("--reservoir", choices=("burgers", "reaction_diffusion", "ks"), default="burgers")
    parser.add_argument("--target-nu", type=float, default=0.05)
    parser.add_argument("--T", type=float, default=1.0)
    parser.add_argument("--rd-nu", type=float, default=1e-3)
    parser.add_argument("--rd-alpha", type=float, default=1.0)
    parser.add_argument("--rd-beta", type=float, default=1.0)
    parser.add_argument("--res-burgers-nu", type=float, default=0.05)
    parser.add_argument("--res-burgers-b", type=float, default=1.0)
    parser.add_argument("--ks-dealias", action="store_true")
    parser.add_argument("--ks-b", type=float, default=1.0)
    parser.add_argument("--ks-eta", type=float, default=1.0)
    parser.add_argument("--ks-kappa", type=float, default=1.0)
    parser.add_argument("--model3-m", type=int, default=256)
    parser.add_argument("--model3-activation", choices=("tanh", "relu", "identity"), default="tanh")
    parser.add_argument("--model3-seed", type=int, default=0)
    parser.add_argument("--model3-weight-scale", type=float, default=0.0)
    parser.add_argument("--model3-bias-scale", type=float, default=1.0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--out-dir", type=str, default="visualizations/model123_1d")
    parser.add_argument("--save-dataset", action="store_true")
    args = parser.parse_args()
    return ExperimentConfig(
        total_samples=args.total_samples,
        ntrain=args.ntrain,
        ntest=args.ntest,
        nx=args.nx,
        dt=args.dt,
        fine_dt=args.fine_dt,
        batch_size=args.batch_size,
        seed=args.seed,
        obs=args.obs,
        J=args.J,
        K=args.K,
        feature_times=args.feature_times,
        reservoir=args.reservoir,
        target_nu=args.target_nu,
        T=args.T,
        rd_nu=args.rd_nu,
        rd_alpha=args.rd_alpha,
        rd_beta=args.rd_beta,
        burgers_nu=args.res_burgers_nu,
        burgers_b=args.res_burgers_b,
        ks_dealias=args.ks_dealias,
        ks_b=args.ks_b,
        ks_eta=args.ks_eta,
        ks_kappa=args.ks_kappa,
        model3_m=args.model3_m,
        model3_activation=args.model3_activation,
        model3_seed=args.model3_seed,
        model3_weight_scale=args.model3_weight_scale,
        model3_bias_scale=args.model3_bias_scale,
        device=args.device,
        dtype=args.dtype,
        out_dir=args.out_dir,
        save_dataset=args.save_dataset,
    )


def main() -> None:
    cfg = parse_args()
    metrics = run_experiment(cfg)
    for key, value in metrics.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
