#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pol.metadata import file_sha256, get_git_info, get_runtime_info
from scripts.suite_common import (
    alpha_ttilde_pairs,
    best_by_validation,
    model123_command,
    parse_floats,
    parse_models,
    read_config_defaults,
    resolve_T,
    row_from_model123_run,
    row_from_zeta_run,
    run_recorded_command,
    safe_tag,
    save_all_formats,
    to_jsonable,
    write_csv,
    write_json,
    zeta_command,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run E2 Burgers-to-Burgers calibration suite.")
    parser.add_argument("--config", default="configs/B0_smoke.json")
    parser.add_argument("--data-file", required=True)
    parser.add_argument("--output-dir", default="outputs/burgers_calibration")
    parser.add_argument("--models", default="model1,model2,model3")
    parser.add_argument("--alpha-values", default="1.0")
    parser.add_argument("--Ttilde-values", default="")
    parser.add_argument("--res-burgers-nu-values", default="0.01")
    parser.add_argument("--res-burgers-b-values", default="1.0")
    parser.add_argument("--zeta-grid", default="1e-8,1e-6,1e-4")
    parser.add_argument("--compute-time-scaled-defect", action="store_true")
    parser.add_argument("--defect-target-nu", type=float, default=None)
    parser.add_argument("--defect-time-quadrature", choices=("trapezoid", "left"), default="trapezoid")
    parser.add_argument("--defect-beta-mode", choices=("zero", "fixed"), default="zero")
    parser.add_argument("--burgers-scheme", choices=("semi_implicit", "split_step", "etdrk4"), default="split_step")
    parser.add_argument("--burgers-dealias", type=int, choices=(0, 1), default=1)
    parser.add_argument("--use-feature-cache", action="store_true")
    parser.add_argument("--refresh-feature-cache", action="store_true")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--require-complete-metadata", action="store_true")
    parser.add_argument("--allow-metadata-mismatch", action="store_true")
    parser.add_argument("--sub", type=int, default=1)
    parser.add_argument("--sim-dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--ridge-dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--standardize-features", type=int, choices=(0, 1), default=0)
    parser.add_argument("--elm-h", type=int, default=1024)
    parser.add_argument("--elm-activation", choices=("tanh", "relu", "identity"), default="tanh")
    parser.add_argument("--elm-seed", type=int, default=0)
    parser.add_argument("--elm-weight-scale", type=float, default=0.0)
    parser.add_argument("--elm-bias-scale", type=float, default=1.0)
    return parser


def coefficient_columns(*, alpha: float, target_nu: float, res_burgers_nu: float, res_burgers_b: float) -> dict[str, float]:
    effective_nu = float(alpha) * float(res_burgers_nu)
    effective_b = float(alpha) * float(res_burgers_b)
    mismatch_nu = float(target_nu) - effective_nu
    mismatch_b = 1.0 - effective_b
    return {
        "effective_nu": effective_nu,
        "effective_b": effective_b,
        "mismatch_nu": mismatch_nu,
        "mismatch_b": mismatch_b,
        "abs_mismatch_nu": abs(mismatch_nu),
        "abs_mismatch_b": abs(mismatch_b),
        "scaled_mismatch_norm": math.sqrt(mismatch_nu * mismatch_nu + mismatch_b * mismatch_b),
    }


def target_nu_from_config(config: dict[str, Any], fallback: float = 0.05) -> float:
    target = config.get("target", {})
    return float(target.get("nu", target.get("target_nu", fallback)))


def _plot_scatter(rows: list[dict[str, Any]], x_key: str, y_key: str, out_path: Path, title: str) -> None:
    data = [row for row in rows if row.get(x_key) is not None and row.get(y_key) is not None and row.get("status") == "ok"]
    if not data:
        return
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.scatter([float(row[x_key]) for row in data], [float(row[y_key]) for row in data], s=24)
    ax.set_xlabel(x_key)
    ax.set_ylabel(y_key)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_all_formats(fig, out_path)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    models = parse_models(args.models)
    config = read_config_defaults(args.config)
    T = resolve_T(config)
    target_nu = float(args.defect_target_nu) if args.defect_target_nu is not None else target_nu_from_config(config)
    pairs = alpha_ttilde_pairs(alpha_values=args.alpha_values, ttilde_values=args.Ttilde_values or None, T=T)
    out_dir = Path(args.output_dir)
    runs_dir = out_dir / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    commands: list[dict[str, Any]] = []

    for model in models:
        for res_nu in parse_floats(args.res_burgers_nu_values):
            for res_b in parse_floats(args.res_burgers_b_values):
                params = {"res_burgers_nu": res_nu, "res_burgers_b": res_b}
                for alpha, Ttilde in pairs:
                    coeffs = coefficient_columns(alpha=alpha, target_nu=target_nu, res_burgers_nu=res_nu, res_burgers_b=res_b)
                    tag = "__".join([model, f"alpha_{safe_tag(alpha)}", f"nu_{safe_tag(res_nu)}", f"b_{safe_tag(res_b)}"])
                    zeta_dir = runs_dir / tag / "zeta_path"
                    run_dir = runs_dir / tag / "selected_run"
                    row = {
                        "phase": "E2",
                        "model": model,
                        "reservoir": "burgers",
                        "alpha": alpha,
                        "Ttilde": Ttilde,
                        "target_nu": target_nu,
                        "res_burgers_nu": res_nu,
                        "res_burgers_b": res_b,
                        "burgers_scheme": args.burgers_scheme,
                        "burgers_dealias": int(args.burgers_dealias),
                        **coeffs,
                    }
                    try:
                        selected_zeta = None
                        if model in {"model2", "model3"}:
                            cmd = zeta_command(args, model=model, reservoir="burgers", Ttilde=Ttilde, params=params, out_dir=zeta_dir)
                            run_recorded_command(
                                name="zeta_path",
                                command=cmd,
                                cwd=REPO_ROOT,
                                commands=commands,
                                dry_run=args.dry_run,
                                log_path=zeta_dir / "suite_command.log",
                            )
                            if not args.dry_run:
                                zeta_row = row_from_zeta_run(zeta_dir)
                                selected_zeta = zeta_row.get("zeta_selected")
                                row.update(zeta_row)
                        if model == "model1" or args.compute_time_scaled_defect:
                            cmd = model123_command(
                                args,
                                model=model,
                                reservoir="burgers",
                                Ttilde=Ttilde,
                                params=params,
                                out_dir=run_dir,
                                zeta=selected_zeta,
                                compute_defect=args.compute_time_scaled_defect,
                            )
                            run_recorded_command(
                                name="model123_selected",
                                command=cmd,
                                cwd=REPO_ROOT,
                                commands=commands,
                                dry_run=args.dry_run,
                                log_path=run_dir / "suite_command.log",
                            )
                            if not args.dry_run:
                                row.update(row_from_model123_run(run_dir))
                                if model == "model1":
                                    row["zeta_selected"] = None
                        row["status"] = "dry_run" if args.dry_run else "ok"
                    except Exception as exc:
                        row.update({"status": "fail", "reason": str(exc)})
                        failures.append(row)
                    rows.append(row)

    ok_rows = [row for row in rows if row.get("status") == "ok"]
    best_rows = [] if args.dry_run else best_by_validation(ok_rows, ["model"])
    preferred = [
        "phase",
        "model",
        "reservoir",
        "alpha",
        "Ttilde",
        "target_nu",
        "res_burgers_nu",
        "res_burgers_b",
        "effective_nu",
        "effective_b",
        "mismatch_nu",
        "mismatch_b",
        "abs_mismatch_nu",
        "abs_mismatch_b",
        "scaled_mismatch_norm",
        "zeta_selected",
        "selection_metric_name",
        "selection_metric_value",
        "train_absL2h",
        "val_absL2h",
        "test_absL2h",
        "test_relL2_mean",
        "test_relL2_agg",
        "delta_scale_rms_abs_l2h",
        "delta_scale_mean_abs_l2h",
        "corr_error_delta_scale_pearson",
        "corr_error_delta_scale_spearman",
        "domain_length",
        "effective_nx",
        "dx",
        "output_dir",
        "status",
    ]
    write_csv(out_dir / "calibration_summary.csv", rows, preferred)
    write_json(out_dir / "calibration_summary.json", {"rows": rows})
    write_csv(out_dir / "best_by_model.csv", best_rows, preferred)
    write_json(out_dir / "best_by_model.json", {"rows": best_rows})
    write_json(out_dir / "commands.json", commands)
    write_json(
        out_dir / "suite_config.json",
        {
            "phase": "E2",
            "args": vars(args),
            "git": get_git_info(REPO_ROOT),
            **get_runtime_info(),
            "config_hash": file_sha256(args.config),
            "data_hash": file_sha256(args.data_file),
            "selection": {"selection_metric": "val_absL2h", "selected_by": "validation"},
            "num_rows": len(rows),
            "num_failures": len(failures),
        },
    )
    if failures:
        write_json(out_dir / "failed_runs.json", failures)
    _plot_scatter(ok_rows, "scaled_mismatch_norm", "test_absL2h", out_dir / "E2_model1_D1_vs_scaled_mismatch", "E2 error vs coefficient mismatch")
    _plot_scatter(ok_rows, "scaled_mismatch_norm", "delta_scale_rms_abs_l2h", out_dir / "E2_delta_scale_vs_scaled_mismatch", "E2 defect vs coefficient mismatch")
    _plot_scatter(ok_rows, "effective_nu", "val_absL2h", out_dir / "E2_best_validation_by_effective_coefficients", "E2 validation by effective nu")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
