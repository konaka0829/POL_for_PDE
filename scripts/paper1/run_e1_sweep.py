#!/usr/bin/env python3
"""Batch runner for Paper 1 E1 spatial-resolution and observation sweeps.

Place this file at scripts/paper1/run_e1_sweep.py in the repository.
It generates one ordinary E1 config per valid (n_tar, n_sur, J), invokes
run_e1.py in separate processes, supports resume/parallel execution, and
collects the main CSV outputs into sweep-level tables.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import copy
import csv
import json
import os
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
RUN_E1 = ROOT / "scripts" / "paper1" / "run_e1.py"


@dataclass(frozen=True, order=True)
class RunSpec:
    n_tar: int
    n_sur: int
    J: int

    @property
    def run_id(self) -> str:
        return f"ntar{self.n_tar}_nsur{self.n_sur}_J{self.J}"

    @property
    def full_observation(self) -> bool:
        return self.J == self.n_sur


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def expand_sweep(spec: dict[str, Any]) -> list[RunSpec]:
    runs: set[RunSpec] = set()

    resolution = spec.get("resolution_sweep")
    if resolution:
        if resolution.get("observation_rule") != "full":
            raise ValueError("resolution_sweep.observation_rule must be 'full'")
        for n_tar in resolution["target_data_nx"]:
            for n_sur in resolution["surrogate_internal_nx"]:
                runs.add(RunSpec(int(n_tar), int(n_sur), int(n_sur)))

    for observation in spec.get("observation_sweeps", []):
        n_tar = int(observation["target_data_nx"])
        n_sur = int(observation["surrogate_internal_nx"])
        for J in observation["observation_dim"]:
            runs.add(RunSpec(n_tar, n_sur, int(J)))

    for explicit in spec.get("explicit_runs", []):
        runs.add(
            RunSpec(
                int(explicit["target_data_nx"]),
                int(explicit["surrogate_internal_nx"]),
                int(explicit["observation_dim"]),
            )
        )

    if not runs:
        raise ValueError("the sweep specification produced no runs")
    return sorted(runs)


def validate_run(run: RunSpec, base_config: dict[str, Any]) -> None:
    if min(run.n_tar, run.n_sur, run.J) <= 0:
        raise ValueError(f"{run.run_id}: dimensions must be positive")
    if run.J > run.n_sur:
        raise ValueError(f"{run.run_id}: J must be <= n_sur")
    if not run.full_observation and run.J < run.n_tar:
        raise ValueError(
            f"{run.run_id}: v24 reduced observation requires J >= n_tar"
        )

    q_values = [int(q) for q in base_config["e1"]["output_dims"]]
    q_max = max(q_values)
    k_max = (q_max - 1) // 2
    if k_max >= min(run.n_tar, run.J) / 2:
        raise ValueError(
            f"{run.run_id}: q_max={q_max} is not representable; "
            f"need k_max={k_max} < min(n_tar,J)/2={min(run.n_tar, run.J)/2}"
        )


def make_run_config(base: dict[str, Any], run: RunSpec) -> dict[str, Any]:
    config = copy.deepcopy(base)
    config["spatial"]["target_data_nx"] = run.n_tar
    config["spatial"]["surrogate_internal_nx"] = run.n_sur
    config["spatial"]["observation_dim"] = run.J
    config["e1"]["require_full_observation"] = run.full_observation
    return config


def summary_passed(output_dir: Path) -> bool:
    path = output_dir / "e1_summary.json"
    if not path.exists():
        return False
    try:
        return load_json(path).get("status") == "pass"
    except Exception:
        return False


def execute_one(
    run: RunSpec,
    *,
    base_config: dict[str, Any],
    e0_dir: Path,
    output_root: Path,
    torch_threads: int,
    overwrite: bool,
    resume: bool,
    skip_plots: bool,
) -> dict[str, Any]:
    config_dir = output_root / "generated_configs"
    run_dir = output_root / "runs" / run.run_id
    config_path = config_dir / f"{run.run_id}.json"
    write_json(config_path, make_run_config(base_config, run))

    if resume and summary_passed(run_dir):
        return {**asdict(run), "run_id": run.run_id, "status": "skipped_passed", "returncode": 0}

    command = [
        sys.executable,
        str(RUN_E1),
        "--config", str(config_path),
        "--e0-dir", str(e0_dir),
        "--output-dir", str(run_dir),
        "--torch-threads", str(torch_threads),
    ]
    if overwrite or (resume and run_dir.exists()):
        command.append("--overwrite")
    if skip_plots:
        command.append("--skip-plots")

    log_dir = output_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{run.run_id}.log"
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
            env=os.environ.copy(),
        )

    status = "pass" if completed.returncode == 0 and summary_passed(run_dir) else "fail"
    return {
        **asdict(run),
        "run_id": run.run_id,
        "status": status,
        "returncode": completed.returncode,
        "config": str(config_path),
        "output_dir": str(run_dir),
        "log": str(log_path),
    }


def collect_outputs(output_root: Path, runs: list[RunSpec]) -> None:
    tables = {
        "selected_results": [],
        "readout_diagnostics": [],
        "noise_summary": [],
    }
    for run in runs:
        run_dir = output_root / "runs" / run.run_id
        if not summary_passed(run_dir):
            continue
        prefix = {
            "run_id": run.run_id,
            "n_tar": run.n_tar,
            "n_sur": run.n_sur,
            "J": run.J,
            "full_observation": run.full_observation,
        }
        for table_name in tables:
            for row in read_csv(run_dir / f"{table_name}.csv"):
                tables[table_name].append({**prefix, **row})

    for name, rows in tables.items():
        write_csv(output_root / f"sweep_{name}.csv", rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Batch Paper 1 E1 sweep")
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--sweep-spec", required=True)
    parser.add_argument("--e0-dir", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--with-per-run-plots", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.jobs <= 0 or args.torch_threads <= 0:
        raise SystemExit("--jobs and --torch-threads must be positive")

    base_config_path = Path(args.base_config).resolve()
    sweep_spec_path = Path(args.sweep_spec).resolve()
    e0_dir = Path(args.e0_dir).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    base_config = load_json(base_config_path)
    sweep_spec = load_json(sweep_spec_path)
    runs = expand_sweep(sweep_spec)
    for run in runs:
        validate_run(run, base_config)

    write_json(
        output_root / "sweep_plan.json",
        {
            "base_config": str(base_config_path),
            "sweep_spec": str(sweep_spec_path),
            "e0_dir": str(e0_dir),
            "num_runs": len(runs),
            "runs": [asdict(run) | {"run_id": run.run_id} for run in runs],
        },
    )

    kwargs = dict(
        base_config=base_config,
        e0_dir=e0_dir,
        output_root=output_root,
        torch_threads=args.torch_threads,
        overwrite=args.overwrite,
        resume=not args.no_resume,
        skip_plots=not args.with_per_run_plots,
    )

    results: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as executor:
        future_map = {
            executor.submit(execute_one, run, **kwargs): run for run in runs
        }
        for future in concurrent.futures.as_completed(future_map):
            run = future_map[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {
                    **asdict(run),
                    "run_id": run.run_id,
                    "status": "exception",
                    "returncode": -1,
                    "error": repr(exc),
                }
            results.append(result)
            print(f"[{result['status']}] {run.run_id}", flush=True)

    results.sort(key=lambda row: row["run_id"])
    write_json(output_root / "sweep_runs.json", results)
    failures = [row for row in results if row["status"] not in {"pass", "skipped_passed"}]
    write_json(output_root / "failed_runs.json", failures)
    collect_outputs(output_root, runs)

    passed = len(results) - len(failures)
    print(f"completed: {passed}/{len(results)} passed or resumed")
    print(f"aggregate: {output_root / 'sweep_selected_results.csv'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
