#!/usr/bin/env python3
"""Generate a portable scientific baseline from a saved passing E1 matrix."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pol.paper1.regression_baseline import (  # noqa: E402
    build_e1_matrix_scientific_baseline,
    write_phase1_scientific_baseline,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix-dir", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    try:
        baseline = build_e1_matrix_scientific_baseline(
            args.matrix_dir.resolve(),
            source_revision=args.source_revision,
        )
        write_phase1_scientific_baseline(
            baseline, args.output, overwrite=args.overwrite
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
