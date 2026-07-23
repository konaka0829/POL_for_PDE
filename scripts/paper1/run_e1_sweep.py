#!/usr/bin/env python3
"""CLI for resumable Paper 1 E1 grid sweeps and aggregate plots."""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pol.paper1.e1_sweep import (
    SweepRun, collect_outputs, expand_sweep, load_json, preflight, summary_passed, write_json,
)
from pol.paper1.e1_sweep_plotting import generate_aggregate_plots

RUN_E1 = ROOT / "scripts/paper1/run_e1.py"
DEFAULT_BASE = ROOT / "configs/paper1_e1_main.json"
DEFAULT_SPEC = ROOT / "configs/paper1_e1_sweep_main.json"
DEFAULT_E0 = ROOT / "outputs_paper1/paper1_e0_main"
DEFAULT_OUTPUT = ROOT / "outputs_paper1/paper1_e1_sweep_extended"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Paper 1 E1 grid sweep")
    parser.add_argument("--base-config", default=str(DEFAULT_BASE))
    parser.add_argument("--sweep-spec", default=str(DEFAULT_SPEC))
    parser.add_argument("--e0-dir", default=str(DEFAULT_E0))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--with-per-run-plots", action="store_true")
    parser.add_argument("--skip-aggregate-plots", action="store_true")
    return parser


def execute_one(
    run: SweepRun, config: dict[str, Any], *, e0_dir: Path, output_root: Path,
    torch_threads: int, overwrite: bool, resume: bool, with_plots: bool,
) -> dict[str, Any]:
    run_dir = output_root / "runs" / run.run_id
    config_path = output_root / "generated_configs" / f"{run.run_id}.json"
    write_json(config_path, config)
    if resume and summary_passed(run_dir):
        return {**run.metadata(), "status": "resumed", "returncode": 0}
    command = [sys.executable, str(RUN_E1), "--config", str(config_path), "--e0-dir", str(e0_dir),
               "--output-dir", str(run_dir), "--torch-threads", str(torch_threads)]
    if overwrite or run_dir.exists():
        command.append("--overwrite")
    if not with_plots:
        command.append("--skip-plots")
    log_path = output_root / "logs" / f"{run.run_id}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                   text=True, check=False, env=os.environ.copy())
    return {**run.metadata(), "status": "pass" if completed.returncode == 0 and summary_passed(run_dir) else "fail",
            "returncode": completed.returncode, "config": str(config_path), "output_dir": str(run_dir),
            "log": str(log_path)}


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.jobs <= 0 or args.torch_threads <= 0:
        parser.error("--jobs and --torch-threads must be positive")
    if args.plot_only and (args.dry_run or args.overwrite or args.no_resume or args.with_per_run_plots):
        parser.error("--plot-only cannot be combined with execution flags")
    base, spec = load_json(Path(args.base_config)), load_json(Path(args.sweep_spec))
    try:
        runs, raw_counts = expand_sweep(spec)
        valid, invalid, configs = preflight(runs, base)
    except ValueError as exc:
        parser.error(str(exc))
    output_root = Path(args.output_root)
    plan = {
        "schema_version": "paper1-e1-sweep-plan-v2", "raw_run_counts": raw_counts,
        "unique_runs": len(runs), "valid_runs": len(valid), "invalid_runs": len(invalid),
        "contains_n_tar_gt_J": any(r.n_tar > r.J for r in valid),
        "contains_n_tar_lt_J": any(r.n_tar < r.J for r in valid),
        "full_observation_runs": sum(r.full_observation for r in valid),
        "runs": [{**r.metadata(), "status": "valid"} for r in valid] + invalid,
    }
    if not args.plot_only:
        write_json(output_root / "sweep_plan.json", plan)
        write_json(output_root / "skipped_invalid_runs.json", invalid)
    print(f"unique valid runs = {len(valid)}")
    print(f"invalid runs = {len(invalid)}")
    print(f"contains n_tar > J = {str(plan['contains_n_tar_gt_J']).lower()}")
    print(f"contains n_tar < J = {str(plan['contains_n_tar_lt_J']).lower()}")
    if invalid and spec.get("invalid_run_policy", "skip") == "error":
        print(json.dumps(invalid, indent=2))
        return 2
    if args.dry_run:
        return 0
    plot_settings = dict(spec.get("aggregate_plots", {}))
    plot_settings.setdefault("q", max(base["e1"]["output_dims"]))
    if args.plot_only:
        manifest = generate_aggregate_plots(output_root, plot_settings)
        return 0 if manifest["status"] == "pass" else 1
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "generated_configs").mkdir(exist_ok=True)
    (output_root / "runs").mkdir(exist_ok=True)
    worker = lambda run: execute_one(run, configs[run.run_id], e0_dir=Path(args.e0_dir),
                                      output_root=output_root, torch_threads=args.torch_threads,
                                      overwrite=args.overwrite, resume=not args.no_resume,
                                      with_plots=args.with_per_run_plots)
    if args.jobs == 1:
        results = [worker(run) for run in valid]
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
            results = list(pool.map(worker, valid))
    write_json(output_root / "sweep_runs.json", results)
    failed = [result for result in results if result["status"] == "fail"]
    write_json(output_root / "failed_runs.json", failed)
    collect_outputs(output_root, valid)
    plot_failed = False
    if plot_settings.get("enabled", True) and not args.skip_aggregate_plots:
        try:
            plot_failed = generate_aggregate_plots(output_root, plot_settings)["status"] != "pass"
        except Exception as exc:
            write_json(output_root / "sweep_plot_manifest.json", {"schema_version": "paper1-e1-sweep-plots-v2",
                       "status": "fail", "reason": str(exc), "plots": []})
            plot_failed = True
    return 1 if failed or plot_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
