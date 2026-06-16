#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model123_burgers_1d import load_data, resolve_domain_length
from pol.metadata import file_sha256, get_command_line, get_git_info, get_runtime_info, normalize_dataset_metadata, to_jsonable
from pol.model123_1d.readouts import FourierDiagonalReadout, evaluate_readout, target_variance_l2h
from scripts.run_zeta_path import apply_config_defaults, ensure_metadata_expectation_defaults


def _zeta_grid(raw: str) -> list[float]:
    return [float(item) for item in raw.split(",") if item.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Estimate Burgers linear headroom with Fourier diagonal readout.")
    parser.add_argument("--config", default="")
    parser.add_argument("--data-file", required=True)
    parser.add_argument("--data-mode", choices=("single_split", "separate_files"), default="single_split")
    parser.add_argument("--train-file", default=None)
    parser.add_argument("--test-file", default=None)
    parser.add_argument("--out-dir", "--output-dir", dest="out_dir", default="outputs/headroom_burgers")
    parser.add_argument("--ntrain", type=int, default=800)
    parser.add_argument("--nval", type=int, default=200)
    parser.add_argument("--ntest", type=int, default=200)
    parser.add_argument("--train-split", type=float, default=0.75)
    parser.add_argument("--shuffle", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-seed", type=int, default=None)
    parser.add_argument("--split-seed", type=int, default=None)
    parser.add_argument("--sub", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--T", type=float, default=1.0)
    parser.add_argument("--target-nu", type=float, default=None)
    parser.add_argument("--Ttilde", type=float, default=0.0)
    parser.add_argument("--dt", type=float, default=1e-2)
    parser.add_argument("--data-dtype", choices=("preserve", "float32", "float64"), default="float32")
    parser.add_argument("--sim-dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--ridge-dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--zeta-grid", default="1e-12,1e-10,1e-8,1e-6,1e-4,1e-2,1e0")
    parser.add_argument("--allow-metadata-mismatch", action="store_true")
    parser.add_argument("--require-complete-metadata", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    apply_config_defaults(parser, args)
    if not hasattr(args, "_config_applied"):
        args._config_applied = {}
    ensure_metadata_expectation_defaults(args)
    args.data_seed = args.seed if args.data_seed is None else args.data_seed
    args.split_seed = args.seed if args.split_seed is None else args.split_seed
    if args.Ttilde <= 0.0:
        args.Ttilde = args.T
    x_train, y_train, x_val, y_val, x_test, y_test, split_meta, dataset_meta, metadata_validation = load_data(args)
    domain_length = resolve_domain_length(args, dataset_meta)
    args.expected_domain_length = domain_length
    effective_nx = int(x_train.shape[1])
    dx = float(domain_length) / float(effective_nx)
    if x_val.shape[0] <= 0:
        raise ValueError("headroom selection requires --nval > 0")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    best = None
    for zeta in _zeta_grid(args.zeta_grid):
        readout = FourierDiagonalReadout(zeta=zeta).fit(x_train.double(), y_train.double())
        pred_val = readout.predict(x_val.double())
        pred_test = readout.predict(x_test.double())
        val = evaluate_readout(pred_val, y_val.double(), domain_length=domain_length)
        test = evaluate_readout(pred_test, y_test.double(), domain_length=domain_length)
        row = {
            "zeta": zeta,
            "val_absL2h": val["absL2h"],
            "test_absL2h": test["absL2h"],
            "test_relL2_mean": test["relL2_mean"],
            "test_relL2_agg": test["relL2_agg"],
            "selected_by_val": False,
        }
        rows.append(row)
        if best is None or row["val_absL2h"] < best["val_absL2h"]:
            best = row
    best["selected_by_val"] = True
    sigma2 = target_variance_l2h(y_test.double(), domain_length=domain_length)
    dlin2 = float(best["test_absL2h"]) ** 2
    summary = {
        "domain_length": domain_length,
        "effective_nx": effective_nx,
        "dx": dx,
        "Dlin2_l2h_convention": "mean squared L2h prediction error with dx=domain_length/effective_nx",
        "sigma_T2_l2h_convention": "target variance in L2h with dx=domain_length/effective_nx",
        "selection": {
            "selection_metric": "val_absL2h",
            "selected_by": "validation",
            "selected_zeta": best["zeta"],
            "legacy_fallback_warning": False,
        },
        "selected_zeta": best["zeta"],
        "test_Dlin2": dlin2,
        "sigma_T2": sigma2,
        "headroom_H": dlin2 / sigma2 if sigma2 > 0 else None,
        "linear_explained_variance": 1.0 - dlin2 / sigma2 if sigma2 > 0 else None,
    }
    with (out_dir / "headroom_zeta_path.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (out_dir / "headroom_summary.json").write_text(json.dumps(to_jsonable(summary), indent=2), encoding="utf-8")
    run_config = {
        "args": vars(args),
        "git": get_git_info(REPO_ROOT),
        **get_runtime_info(),
        "command_line": get_command_line(),
        "data_file": args.data_file,
        "data_sha256": file_sha256(args.data_file),
        "dataset_metadata": normalize_dataset_metadata(dataset_meta),
        "metadata_validation": metadata_validation,
        "split": split_meta,
        "grid": {
            "dataset_nx": int(split_meta.get("dataset_nx", split_meta.get("raw_nx", effective_nx * int(args.sub)))),
            "raw_nx": int(split_meta.get("raw_nx", split_meta.get("dataset_nx", effective_nx * int(args.sub)))),
            "effective_nx": effective_nx,
            "sub": int(args.sub),
            "domain_length": domain_length,
            "dx": dx,
        },
        "selection": summary["selection"],
        "metrics": {**summary, "best_by_val": best},
        "readout": {
            "ridge_parameter_name": "zeta",
            "domain_length": domain_length,
            "effective_nx": effective_nx,
            "dx": dx,
        },
        "dtype": {"data_dtype": args.data_dtype, "sim_dtype": args.sim_dtype, "ridge_dtype": args.ridge_dtype},
    }
    (out_dir / "run_config.json").write_text(json.dumps(to_jsonable(run_config), indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
