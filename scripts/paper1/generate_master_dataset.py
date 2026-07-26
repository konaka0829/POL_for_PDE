#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pol.runtime.recipe import (
    RecipeInvocation,
    RecipeUsageError,
    numerical_thread_scope,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a Phase 1 Paper 1 master dataset"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--no-target",
        action="store_true",
        help="save only master initial conditions",
    )
    parser.add_argument(
        "--master-initial-conditions",
        help="validated E0 master_initial_conditions.pt archive",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    invocation_arguments = sys.argv[1:] if argv is None else argv
    invocation = RecipeInvocation(
        repo_root=REPO_ROOT,
        working_directory=Path.cwd(),
        command=(
            sys.executable,
            str(Path(__file__).relative_to(REPO_ROOT)),
            *invocation_arguments,
        ),
        torch_threads=1,
    )
    try:
        with numerical_thread_scope(1):
            from pol.paper1.recipes.master_dataset import (
                run_master_dataset_generation,
            )

            result = run_master_dataset_generation(
                Path(args.config),
                Path(args.output_dir),
                overwrite=args.overwrite,
                generate_target=not args.no_target,
                master_initial_conditions=(
                    None
                    if args.master_initial_conditions is None
                    else Path(args.master_initial_conditions)
                ),
                invocation=invocation,
            )
    except RecipeUsageError as exc:
        parser.error(str(exc))
    print(json.dumps(result.console_payload, sort_keys=True, allow_nan=False))
    return result.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
