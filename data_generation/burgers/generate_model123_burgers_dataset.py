from __future__ import annotations

import argparse

from pol.model123_1d import DatasetConfig, build_dataset, save_dataset_bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate fixed Burgers dataset for Model 1-3 experiments")
    parser.add_argument("--out-file", required=True)
    parser.add_argument("--total-samples", type=int, default=1200)
    parser.add_argument("--ntrain", type=int, default=1000)
    parser.add_argument("--ntest", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--nx", type=int, default=256)
    parser.add_argument("--nu", type=float, default=0.05)
    parser.add_argument("--T", type=float, default=1.0)
    parser.add_argument("--dt", type=float, default=1e-3)
    parser.add_argument("--fine-dt", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = DatasetConfig(
        total_samples=args.total_samples,
        ntrain=args.ntrain,
        ntest=args.ntest,
        seed=args.seed,
        nx=args.nx,
        target_nu=args.nu,
        T=args.T,
        dt=args.dt,
        fine_dt=args.fine_dt,
        batch_size=args.batch_size,
        dtype=args.dtype,
    )
    bundle = build_dataset(cfg)
    save_dataset_bundle(bundle, args.out_file)
    print(f"saved dataset: {args.out_file}")
    print(f"train: {tuple(bundle.u0_train.shape)} -> {tuple(bundle.y_train.shape)}")
    print(f"test: {tuple(bundle.u0_test.shape)} -> {tuple(bundle.y_test.shape)}")


if __name__ == "__main__":
    main()
