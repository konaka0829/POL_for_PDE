#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pol.metadata import file_sha256, get_git_info, get_runtime_info
from scripts.suite_common import (
    alpha_ttilde_pairs,
    best_by_validation,
    parse_models,
    parse_reservoirs,
    read_config_defaults,
    resolve_T,
    reservoir_grid,
    row_from_zeta_run,
    run_recorded_command,
    safe_tag,
    SuiteJob,
    SuiteJobResult,
    to_jsonable,
    run_suite_jobs,
    write_csv,
    write_json,
    zeta_command,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run E1 baseline suite for static, heat, and advection reservoirs.")
    parser.add_argument("--config", default="configs/B0_smoke.json")
    parser.add_argument("--data-file", required=True)
    parser.add_argument("--output-dir", default="outputs/baseline_suite")
    parser.add_argument("--models", default="model2,model3")
    parser.add_argument("--reservoirs", default="static,heat,advection")
    parser.add_argument("--zeta-grid", default="1e-8,1e-6,1e-4")
    parser.add_argument("--alpha-values", default="1.0")
    parser.add_argument("--Ttilde-values", default="")
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
    parser.add_argument("--heat-nu-values", default="0.01")
    parser.add_argument("--advection-c-values", default="1.0")
    parser.add_argument("--res-burgers-nu-values", default="0.01")
    parser.add_argument("--res-burgers-b-values", default="1.0")
    parser.add_argument("--rd-nu-values", default="0.001")
    parser.add_argument("--rd-alpha-values", default="1.0")
    parser.add_argument("--rd-beta-values", default="1.0")
    parser.add_argument("--ks-b-values", default="1.0")
    parser.add_argument("--ks-eta-values", default="1.0")
    parser.add_argument("--ks-kappa-values", default="1.0")
    parser.add_argument("--burgers-scheme", choices=("semi_implicit", "split_step", "etdrk4"), default="split_step")
    parser.add_argument("--burgers-dealias", type=int, choices=(0, 1), default=1)
    parser.add_argument("--ks-dealias", action="store_true")
    return parser


def _row_base(*, args: argparse.Namespace, model: str, reservoir: str, alpha: float, Ttilde: float, params: dict[str, Any], child_dir: Path) -> dict[str, Any]:
    row = {
        "phase": "E1",
        "model": model,
        "reservoir": reservoir,
        "alpha": alpha,
        "Ttilde": Ttilde,
        "output_dir": str(child_dir),
        **params,
    }
    if model == "model3":
        row.update(
            {
                "elm_h": args.elm_h,
                "elm_activation": args.elm_activation,
                "elm_seed": args.elm_seed,
                "elm_weight_scale": getattr(args, "elm_weight_scale", None),
                "elm_bias_scale": getattr(args, "elm_bias_scale", None),
            }
        )
    return row


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    models = parse_models(args.models, allowed={"model2", "model3"})
    reservoirs = parse_reservoirs(args.reservoirs)
    config = read_config_defaults(args.config)
    T = resolve_T(config)
    pairs = alpha_ttilde_pairs(alpha_values=args.alpha_values, ttilde_values=args.Ttilde_values or None, T=T)
    out_dir = Path(args.output_dir)
    runs_dir = out_dir / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs: list[SuiteJob] = []
    seen_dirs: set[Path] = set()
    index = 0

    for model in models:
        for reservoir in reservoirs:
            for params in reservoir_grid(reservoir, args):
                for alpha, Ttilde in pairs:
                    tag_parts = [model, reservoir, f"alpha_{safe_tag(alpha)}"] + [f"{k}_{safe_tag(v)}" for k, v in sorted(params.items())]
                    child_dir = runs_dir / "__".join(tag_parts)
                    if child_dir in seen_dirs:
                        raise ValueError(f"duplicate suite output directory: {child_dir}")
                    seen_dirs.add(child_dir)
                    row = _row_base(args=args, model=model, reservoir=reservoir, alpha=alpha, Ttilde=Ttilde, params=params, child_dir=child_dir)
                    cmd = zeta_command(args, model=model, reservoir=reservoir, Ttilde=Ttilde, params=params, out_dir=child_dir)
                    label = f"{model}/{reservoir}/alpha={alpha}"

                    def run_job(index=index, row=row, cmd=cmd, child_dir=child_dir) -> SuiteJobResult:
                        commands_local: list[dict[str, Any]] = []
                        failures_local: list[dict[str, Any]] = []
                        row = dict(row)
                        try:
                            run_recorded_command(
                                name="zeta_path",
                                command=cmd,
                                cwd=REPO_ROOT,
                                commands=commands_local,
                                dry_run=args.dry_run,
                                log_path=child_dir / "suite_command.log",
                            )
                            if args.dry_run:
                                row.update({"status": "dry_run"})
                            else:
                                row.update(row_from_zeta_run(child_dir))
                                row.update({"status": "ok"})
                        except Exception as exc:
                            row.update({"status": "fail", "reason": str(exc)})
                            failures_local.append(dict(row))
                        return SuiteJobResult(index=index, row=row, commands=commands_local, failures=failures_local)

                    jobs.append(SuiteJob(index=index, label=label, run=run_job))
                    index += 1

    results = run_suite_jobs(jobs=jobs, max_workers=args.max_workers, progress_label="E1")
    rows = [result.row for result in results]
    commands = [command for result in results for command in result.commands]
    failures = [failure for result in results for failure in result.failures]

    best_rows = [] if args.dry_run else best_by_validation([row for row in rows if row.get("status") == "ok"], ["model", "reservoir"])
    preferred = [
        "phase",
        "model",
        "reservoir",
        "alpha",
        "Ttilde",
        "zeta_selected",
        "selection_metric_name",
        "selection_metric_value",
        "train_absL2h",
        "val_absL2h",
        "test_absL2h",
        "train_relL2_mean",
        "val_relL2_mean",
        "test_relL2_mean",
        "test_relL2_agg",
        "W_fro_norm",
        "W_l2h_hs_norm",
        "d_eff",
        "cond_zeta",
        "domain_length",
        "effective_nx",
        "dx",
        "data_hash",
        "split_hash",
        "output_dir",
        "status",
    ]
    write_csv(out_dir / "baseline_summary.csv", rows, preferred)
    write_json(out_dir / "baseline_summary.json", {"rows": rows})
    write_csv(out_dir / "best_by_model_reservoir.csv", best_rows, preferred)
    write_json(out_dir / "best_by_model_reservoir.json", {"rows": best_rows})
    write_json(out_dir / "commands.json", commands)
    if failures:
        write_json(out_dir / "failed_runs.json", failures)
    suite_config = {
        "phase": "E1",
        "args": vars(args),
        "git": get_git_info(REPO_ROOT),
        **get_runtime_info(),
        "config_hash": file_sha256(args.config),
        "data_hash": file_sha256(args.data_file),
        "selection": {"selection_metric": "val_absL2h", "selected_by": "validation"},
        "num_rows": len(rows),
        "num_failures": len(failures),
    }
    write_json(out_dir / "suite_config.json", to_jsonable(suite_config))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
