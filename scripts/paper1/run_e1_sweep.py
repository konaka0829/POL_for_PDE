#!/usr/bin/env python3
"""Deprecated compatibility CLI backed by the generic matrix executor."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pol.paper1.e1_sweep_plotting import generate_aggregate_plots
from pol.workflow.matrix import execute_matrix_run, matrix_plan_to_dict
from pol.workflow.matrix_spec import MatrixRunSpec


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


def _translate_experiment(raw: dict[str, object]) -> dict[str, object]:
    fixed = {
        f"spatial.{key}": value
        for key, value in dict(raw.get("fixed", {})).items()
    }
    grid_raw = dict(raw.get("grid", {}))
    grid = [
        {"path": f"spatial.{key}", "values": grid_raw[key]}
        for key in sorted(grid_raw)
    ]
    copies = (
        {
            "spatial.observation_dim":
                "spatial.surrogate_internal_nx"
        }
        if raw.get("observation_rule") == "full"
        else {}
    )
    return {
        "name": raw["name"],
        "fixed": fixed,
        "grid": grid,
        "copy": copies,
    }


def _compatibility_spec(args: argparse.Namespace) -> tuple[MatrixRunSpec, dict]:
    source = Path(args.sweep_spec).resolve()
    old = json.loads(source.read_text(encoding="utf-8"))
    if old.get("schema_version") != "paper1-e1-sweep-v2":
        raise ValueError("legacy wrapper requires paper1-e1-sweep-v2")
    output = Path(args.output_root).resolve()
    if output.name in {"", ".", ".."}:
        raise ValueError("unsafe --output-root")
    experiments = tuple(
        _translate_experiment(item) for item in old.get("experiments", [])
    )
    explicit = tuple(
        {
            f"spatial.{key}": value
            for key, value in item.items()
        }
        for item in old.get("explicit_runs", [])
    )
    spec = MatrixRunSpec(
        "paper1-matrix-run-v1",
        output.name,
        output.parent,
        "e1",
        Path(args.base_config).resolve(),
        Path(args.base_config).resolve(),
        old.get("invalid_run_policy", "skip"),
        experiments,
        explicit,
        args.jobs,
        args.torch_threads,
        not args.no_resume,
        args.with_per_run_plots,
        "paper1_e1_resolution_v1",
        source,
        old,
    )
    return spec, old


def _copy_legacy_aggregates(output: Path) -> None:
    for name in (
        "sweep_selected_results.csv",
        "sweep_readout_diagnostics.csv",
        "sweep_noise_summary.csv",
    ):
        shutil.copy2(output / "aggregate" / name, output / name)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.jobs <= 0 or args.torch_threads <= 0:
        build_parser().error("--jobs and --torch-threads must be positive")
    if args.plot_only and (
        args.dry_run
        or args.overwrite
        or args.no_resume
        or args.with_per_run_plots
    ):
        build_parser().error("--plot-only cannot be combined with execution flags")
    print(
        "warning: run_e1_sweep.py is deprecated; use "
        "pol run <paper1-matrix-run-v1 spec>",
        file=sys.stderr,
    )
    try:
        spec, old = _compatibility_spec(args)
        output = spec.run_dir
        settings = dict(old.get("aggregate_plots", {}))
        if args.plot_only:
            manifest = generate_aggregate_plots(output, settings)
            return 0 if manifest["status"] == "pass" else 1
        plan = matrix_plan_to_dict(spec, repo_root=ROOT)
        print(f"unique valid runs = {plan['unique_valid_cells']}")
        print(f"invalid runs = {len(plan['invalid_runs'])}")
        print(
            "contains n_tar > J = "
            f"{str(plan['contains_n_tar_gt_J']).lower()}"
        )
        print(
            "contains n_tar < J = "
            f"{str(plan['contains_n_tar_lt_J']).lower()}"
        )
        if args.dry_run:
            return 0
        code = execute_matrix_run(
            spec,
            repo_root=ROOT,
            force=args.overwrite,
            existing_e0_dir=Path(args.e0_dir),
        )
        _copy_legacy_aggregates(output)
        if code == 0 and settings.get("enabled", True) and not args.skip_aggregate_plots:
            manifest = generate_aggregate_plots(output, settings)
            if manifest["status"] != "pass":
                return 1
        return code
    except (OSError, ValueError) as exc:
        print(f"run_e1_sweep.py: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
