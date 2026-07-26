#!/usr/bin/env python3
"""Thin CLI for Paper 1 E2."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pol.runtime.recipe import RecipeInvocation, RecipeUsageError, numerical_thread_scope


def parser() -> argparse.ArgumentParser:
    """Build the legacy-compatible E2 argument parser."""
    value = argparse.ArgumentParser(description="Paper 1 E2 parameter/time sweeps")
    value.add_argument("--config", required=True)
    value.add_argument("--e0-dir")
    value.add_argument("--dataset-dir")
    value.add_argument("--output-dir")
    modes = value.add_mutually_exclusive_group()
    modes.add_argument("--overwrite", action="store_true")
    modes.add_argument("--resume", action="store_true")
    value.add_argument("--dry-run-cost", action="store_true")
    value.add_argument("--skip-plots", action="store_true")
    value.add_argument("--torch-threads", type=int, default=1)
    value.add_argument("--batch-size", type=int, default=64)
    return value


def main(argv: list[str] | None = None) -> int:
    """Parse legacy CLI arguments and invoke the import-safe E2 recipe."""
    args = parser().parse_args(argv)
    if args.torch_threads <= 0 or args.batch_size <= 0:
        parser().error("--torch-threads and --batch-size must be positive")
    invocation_arguments = sys.argv[1:] if argv is None else argv
    script_name = sys.argv[0] if argv is None else str(Path(__file__))
    command = (sys.executable, script_name, *invocation_arguments)
    invocation = RecipeInvocation(
        repo_root=ROOT,
        working_directory=Path.cwd(),
        command=command,
        torch_threads=args.torch_threads,
    )
    try:
        with numerical_thread_scope(args.torch_threads):
            from pol.paper1.recipes.surrogate_parameter_time import (
                build_surrogate_parameter_time_cost_summary,
                run_surrogate_parameter_time,
            )

            if args.dry_run_cost:
                payload = build_surrogate_parameter_time_cost_summary(Path(args.config))
                print(json.dumps(payload, indent=2, sort_keys=True))
                return 0
            if not args.e0_dir or not args.dataset_dir or not args.output_dir:
                parser().error(
                    "--e0-dir, --dataset-dir, and --output-dir are required "
                    "unless --dry-run-cost is used")
            result = run_surrogate_parameter_time(
                Path(args.config),
                Path(args.e0_dir),
                Path(args.dataset_dir),
                Path(args.output_dir),
                overwrite=args.overwrite,
                resume=args.resume,
                skip_plots=args.skip_plots,
                batch_size=args.batch_size,
                invocation=invocation,
            )
    except RecipeUsageError as exc:
        parser().error(str(exc))
    print(json.dumps(result.console_payload, sort_keys=True, allow_nan=False))
    return result.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
