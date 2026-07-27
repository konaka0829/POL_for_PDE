"""Test-only subprocess driver for import-safe Paper 1 recipes.

The public interface is ``pol run``.  This helper exists solely for
same-runtime direct-recipe parity and failure-contract tests.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pol.runtime.recipe import RecipeInvocation, RecipeUsageError


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("recipe", choices=("e0", "dataset", "e1", "e2"))
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--e0-dir")
    parser.add_argument("--dataset-dir")
    parser.add_argument("--master-initial-conditions")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-target", action="store_true")
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    return parser


def main() -> int:
    args = _parser().parse_args()
    invocation = RecipeInvocation(
        command=("test_direct_recipe", args.recipe),
        repo_root=ROOT,
        working_directory=ROOT,
        torch_threads=args.torch_threads,
    )
    try:
        if args.recipe == "e0":
            from pol.paper1.recipes.foundation_validation import (
                run_foundation_validation,
            )

            result = run_foundation_validation(
                Path(args.config),
                Path(args.output_dir),
                overwrite=args.overwrite,
                invocation=invocation,
            )
        elif args.recipe == "dataset":
            from pol.paper1.recipes.master_dataset import (
                run_master_dataset_generation,
            )

            result = run_master_dataset_generation(
                Path(args.config),
                Path(args.output_dir),
                overwrite=args.overwrite,
                generate_target=not args.no_target,
                master_initial_conditions=(
                    Path(args.master_initial_conditions)
                    if args.master_initial_conditions
                    else None
                ),
                invocation=invocation,
            )
        elif args.recipe == "e1":
            from pol.paper1.recipes.heat_calibration import run_heat_calibration

            result = run_heat_calibration(
                Path(args.config),
                Path(args.e0_dir),
                Path(args.output_dir),
                overwrite=args.overwrite,
                skip_plots=args.skip_plots,
                invocation=invocation,
            )
        else:
            from pol.paper1.recipes.surrogate_parameter_time import (
                run_surrogate_parameter_time,
            )

            result = run_surrogate_parameter_time(
                Path(args.config),
                Path(args.e0_dir),
                Path(args.dataset_dir),
                Path(args.output_dir),
                overwrite=args.overwrite,
                resume=False,
                skip_plots=args.skip_plots,
                batch_size=args.batch_size,
                invocation=invocation,
            )
        return result.exit_code
    except RecipeUsageError as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
