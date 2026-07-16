#!/usr/bin/env python3
"""Legacy pre-paper suite; this is not the current Paper 1 E0 acceptance test.

Use ``scripts/paper1/run_e0.py`` for the current E0 definition.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pol.metadata import file_sha256, get_git_info, get_runtime_info
from scripts.suite_common import load_json, read_config_defaults, resolve_T, run_recorded_command, write_csv, write_json, zeta_command


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run E0 B0 smoke suite.")
    parser.add_argument("--config", default="configs/B0_smoke.json")
    parser.add_argument("--output-dir", default="outputs/e0_smoke")
    parser.add_argument("--data-file", default="")
    parser.add_argument("--use-feature-cache", action="store_true")
    parser.add_argument("--refresh-feature-cache", action="store_true")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _args_for_zeta(args: argparse.Namespace, data_file: Path) -> argparse.Namespace:
    values = vars(args).copy()
    values.update(
        {
            "data_file": str(data_file),
            "zeta_grid": "1e-8,1e-6,1e-4",
            "sub": 1,
            "sim_dtype": "float32",
            "ridge_dtype": "float64",
            "standardize_features": 0,
            "elm_h": 128,
            "elm_activation": "tanh",
            "elm_seed": 0,
            "allow_metadata_mismatch": False,
            "require_complete_metadata": False,
            "heat_nu_values": "0.01",
            "advection_c_values": "1.0",
            "res_burgers_nu_values": "0.01",
            "res_burgers_b_values": "1.0",
            "rd_nu_values": "0.001",
            "rd_alpha_values": "1.0",
            "rd_beta_values": "1.0",
            "ks_b_values": "1.0",
            "ks_eta_values": "1.0",
            "ks_kappa_values": "1.0",
            "burgers_scheme": "split_step",
            "burgers_dealias": 0,
            "ks_dealias": False,
        }
    )
    return argparse.Namespace(**values)


def _summary_row(name: str, status: str, output_dir: Path, metrics: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"task": name, "status": status, "output_dir": str(output_dir), **(metrics or {})}


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    out_dir = Path(args.output_dir)
    data_dir = out_dir / "data"
    runs_dir = out_dir / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    data_file = Path(args.data_file) if args.data_file else data_dir / "burgers_model123.pt"
    config = read_config_defaults(args.config)
    T = resolve_T(config)
    target_nu = float(config.get("target", {}).get("nu", config.get("target", {}).get("target_nu", 0.01)))
    commands: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    if not args.data_file:
        generate_cmd = [
            str(args.python),
            "scripts/generate_burgers_1d.py",
            "--config",
            str(args.config),
            "--format",
            "pt",
            "--output-dir",
            str(data_dir),
        ]
        try:
            run_recorded_command(
                name="generate_b0_dataset",
                command=generate_cmd,
                cwd=REPO_ROOT,
                commands=commands,
                dry_run=args.dry_run,
                log_path=runs_dir / "generate_dataset.log",
            )
            rows.append(_summary_row("generate_b0_dataset", "dry_run" if args.dry_run else "ok", data_dir))
        except Exception as exc:
            row = _summary_row("generate_b0_dataset", "fail", data_dir, {"reason": str(exc)})
            rows.append(row)
            failures.append(row)

    zeta_args = _args_for_zeta(args, data_file)
    for model in ("model2", "model3"):
        child_dir = runs_dir / f"zeta_static_{model}"
        try:
            cmd = zeta_command(zeta_args, model=model, reservoir="static", Ttilde=T, params={}, out_dir=child_dir)
            run_recorded_command(
                name=f"zeta_static_{model}",
                command=cmd,
                cwd=REPO_ROOT,
                commands=commands,
                dry_run=args.dry_run,
                log_path=child_dir / "suite_command.log",
            )
            metrics = {}
            if not args.dry_run:
                best = load_json(child_dir / "best_by_val.json")["best_by_val"]
                metrics = {"selected_zeta": best.get("zeta"), "val_absL2h": best.get("val_absL2h"), "test_absL2h": best.get("test_absL2h")}
            rows.append(_summary_row(f"zeta_static_{model}", "dry_run" if args.dry_run else "ok", child_dir, metrics))
        except Exception as exc:
            row = _summary_row(f"zeta_static_{model}", "fail", child_dir, {"reason": str(exc)})
            rows.append(row)
            failures.append(row)

    headroom_dir = runs_dir / "headroom"
    headroom_cmd = [
        str(args.python),
        "scripts/run_headroom_burgers.py",
        "--config",
        str(args.config),
        "--data-file",
        str(data_file),
        "--zeta-grid",
        "1e-8,1e-6,1e-4",
        "--output-dir",
        str(headroom_dir),
    ]
    try:
        run_recorded_command(
            name="headroom",
            command=headroom_cmd,
            cwd=REPO_ROOT,
            commands=commands,
            dry_run=args.dry_run,
            log_path=headroom_dir / "suite_command.log",
        )
        metrics = {}
        if not args.dry_run:
            summary = load_json(headroom_dir / "headroom_summary.json")
            metrics = {"selected_zeta": summary.get("selected_zeta"), "headroom_H": summary.get("headroom_H")}
        rows.append(_summary_row("headroom", "dry_run" if args.dry_run else "ok", headroom_dir, metrics))
    except Exception as exc:
        row = _summary_row("headroom", "fail", headroom_dir, {"reason": str(exc)})
        rows.append(row)
        failures.append(row)

    defect_dir = runs_dir / "model1_burgers_matching_defect"
    defect_cmd = [
        str(args.python),
        "model123_burgers_1d.py",
        "--config",
        str(args.config),
        "--data-file",
        str(data_file),
        "--model",
        "model1",
        "--reservoir",
        "burgers",
        "--res-burgers-nu",
        str(target_nu),
        "--res-burgers-b",
        "1.0",
        "--burgers-dealias",
        "0",
        "--compute-time-scaled-defect",
        "--out-dir",
        str(defect_dir),
    ]
    try:
        run_recorded_command(
            name="model1_burgers_matching_defect",
            command=defect_cmd,
            cwd=REPO_ROOT,
            commands=commands,
            dry_run=args.dry_run,
            log_path=defect_dir / "suite_command.log",
        )
        metrics = {}
        if not args.dry_run:
            defect = load_json(defect_dir / "time_scaled_defect_metrics.json")
            metrics = {"delta_scale_rms_abs_l2h": defect.get("delta_scale_rms_abs_l2h"), "max_abs_difference_model1_D1": defect.get("max_abs_difference_model1_D1")}
        rows.append(_summary_row("model1_burgers_matching_defect", "dry_run" if args.dry_run else "ok", defect_dir, metrics))
    except Exception as exc:
        row = _summary_row("model1_burgers_matching_defect", "fail", defect_dir, {"reason": str(exc)})
        rows.append(row)
        failures.append(row)

    write_csv(out_dir / "e0_smoke_summary.csv", rows)
    write_json(out_dir / "e0_smoke_summary.json", {"rows": rows, "num_failures": len(failures)})
    write_json(out_dir / "commands.json", commands)
    write_json(
        out_dir / "environment.json",
        {
            "git": get_git_info(REPO_ROOT),
            **get_runtime_info(),
            "config_path": str(args.config),
            "config_hash": file_sha256(args.config),
            "data_file": str(data_file),
            "data_hash": file_sha256(data_file),
            "dry_run": bool(args.dry_run),
        },
    )
    if failures:
        write_json(out_dir / "failed_runs.json", failures)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
