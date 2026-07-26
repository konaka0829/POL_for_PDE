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
        description="Run the current Paper 1 E0 acceptance gate"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--overwrite", action="store_true")
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
            from pol.paper1.recipes.foundation_validation import (
                run_foundation_validation,
            )

            result = run_foundation_validation(
                Path(args.config),
                Path(args.output_dir),
                overwrite=args.overwrite,
                invocation=invocation,
            )
    except RecipeUsageError as exc:
        parser.error(str(exc))
    print(json.dumps(result.console_payload, sort_keys=True, allow_nan=False))
    return result.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
