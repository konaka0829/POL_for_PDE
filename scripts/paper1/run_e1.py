#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pol.runtime.recipe import (
    RecipeInvocation,
    RecipeUsageError,
    numerical_thread_scope,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Paper 1 E1 heat calibration")
    parser.add_argument("--config", required=True)
    parser.add_argument("--e0-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--torch-threads", type=int, default=1)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.torch_threads <= 0:
        parser.error("--torch-threads must be a positive integer")
    invocation_arguments = sys.argv[1:] if argv is None else argv
    script_name = sys.argv[0] if argv is None else str(Path(__file__))
    invocation = RecipeInvocation(
        repo_root=ROOT,
        working_directory=Path.cwd(),
        command=(sys.executable, script_name, *invocation_arguments),
        torch_threads=args.torch_threads,
    )
    try:
        with numerical_thread_scope(args.torch_threads):
            from pol.paper1.recipes.heat_calibration import run_heat_calibration

            result = run_heat_calibration(
                Path(args.config),
                Path(args.e0_dir),
                Path(args.output_dir),
                overwrite=args.overwrite,
                skip_plots=args.skip_plots,
                invocation=invocation,
            )
    except RecipeUsageError as exc:
        parser.error(str(exc))
    print(json.dumps(result.console_payload, sort_keys=True, allow_nan=False))
    return result.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
