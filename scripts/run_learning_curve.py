#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from pol.metadata import get_command_line, get_git_info, get_runtime_info, to_jsonable
from run_zeta_path import apply_config_defaults, build_parser as build_zeta_parser, ensure_metadata_expectation_defaults, run_zeta_path


def parse_ints(raw: str) -> list[int]:
    values = [int(item) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("--train-sizes must contain at least one value")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = build_zeta_parser()
    parser.description = "Run a learning curve with fixed validation/test split and validation-selected zeta."
    parser.add_argument("--train-sizes", required=True)
    parser.set_defaults(output_dir="outputs/learning_curve")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    apply_config_defaults(parser, args)
    if not hasattr(args, "_config_applied"):
        args._config_applied = {}
    ensure_metadata_expectation_defaults(args)
    train_sizes = parse_ints(args.train_sizes)
    args.ntrain = max(max(train_sizes), int(args.ntrain))
    rows = []
    details = {}
    last_summary = None
    for ntrain in train_sizes:
        zeta_rows, summary = run_zeta_path(args, train_limit=ntrain)
        last_summary = summary
        best = summary["best_by_val"]
        rows.append(
            {
                "ntrain": ntrain,
                "selected_zeta": best["zeta"],
                "train_absL2h": best["train_absL2h"],
                "val_absL2h": best["val_absL2h"],
                "test_absL2h": best["test_absL2h"],
                "W_fro_norm": best["W_fro_norm"],
                "d_eff": best["d_eff"],
                "cond_zeta": best["cond_zeta"],
            }
        )
        details[str(ntrain)] = {"zeta_path": zeta_rows, "summary": summary}

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "learning_curve.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (out_dir / "learning_curve.json").write_text(json.dumps(to_jsonable({"rows": rows, "details": details}), indent=2), encoding="utf-8")
    run_config = {
        "args": vars(args),
        "git": get_git_info(REPO_ROOT),
        **get_runtime_info(),
        "command_line": get_command_line(),
        "selection": {"selection_metric": "val_absL2h", "selected_by": "validation"},
        "split": last_summary.get("split") if last_summary else None,
        "grid": last_summary.get("grid") if last_summary else None,
        "domain_length": last_summary.get("domain_length") if last_summary else None,
        "effective_nx": last_summary.get("effective_nx") if last_summary else None,
        "dx": last_summary.get("dx") if last_summary else None,
        "dataset_metadata": last_summary.get("dataset_metadata") if last_summary else None,
        "metadata_validation": last_summary.get("metadata_validation") if last_summary else None,
        "dtype": {
            "data_dtype": args.data_dtype,
            "sim_dtype": args.sim_dtype,
            "ridge_dtype": args.ridge_dtype,
        },
        "metrics": {
            "l2h_convention": "dx=sum_weight_with_dx_L_over_effective_nx",
            "domain_length": last_summary.get("domain_length") if last_summary else None,
            "effective_nx": last_summary.get("effective_nx") if last_summary else None,
            "dx": last_summary.get("dx") if last_summary else None,
            "rows": rows,
        },
    }
    (out_dir / "run_config.json").write_text(json.dumps(to_jsonable(run_config), indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
