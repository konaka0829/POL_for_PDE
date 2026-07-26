#!/usr/bin/env python3
"""Generate a semantic baseline from already-completed Paper 1 artifacts."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pol.paper1.regression_baseline import (
    build_phase1_scientific_baseline,
    write_phase1_scientific_baseline,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the development-only baseline generator CLI."""
    parser = argparse.ArgumentParser(
        description="Generate a semantic baseline from passing saved artifacts"
    )
    parser.add_argument("--e0-dir", required=True, type=Path)
    parser.add_argument("--e1-dir", required=True, type=Path)
    parser.add_argument("--e2-dir", required=True, type=Path)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Validate inputs and write one deterministic baseline JSON file."""
    args = build_parser().parse_args(argv)
    try:
        baseline = build_phase1_scientific_baseline(
            args.e0_dir.resolve(),
            args.e1_dir.resolve(),
            args.e2_dir.resolve(),
            source_revision=args.source_revision,
        )
        write_phase1_scientific_baseline(
            baseline, args.output.resolve(), overwrite=args.overwrite
        )
    except (OSError, TypeError, ValueError) as exc:
        print(f"baseline generator: error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
