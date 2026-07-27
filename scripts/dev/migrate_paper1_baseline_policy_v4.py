"""Explicit metadata-only migration of Paper 1 semantic baseline policies."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pol.paper1.regression_baseline import (
    BASELINE_SCHEMA_VERSION,
    MATRIX_BASELINE_SCHEMA_VERSION,
    _comparison_policy,
)
from pol.paper1.protocols import BASELINE_GENERATOR_VERSION


def migrate(value: dict[str, object]) -> dict[str, object]:
    """Replace policy/version metadata without changing expected records."""
    migrated = dict(value)
    schema = migrated.get("schema_version")
    if schema == "paper1-phase1-scientific-baseline-v3":
        record = {
            key: migrated[key]
            for key in ("e0", "e1", "e2")
        }
        migrated["schema_version"] = BASELINE_SCHEMA_VERSION
        migrated["generator_version"] = BASELINE_GENERATOR_VERSION
        migrated["comparison_policy"] = _comparison_policy(record)
    elif schema == "paper1-e1-matrix-smoke-baseline-v2":
        record = migrated["record"]
        migrated["schema_version"] = MATRIX_BASELINE_SCHEMA_VERSION
        migrated["generator_version"] = BASELINE_GENERATOR_VERSION
        migrated["comparison_policy"] = _comparison_policy(record)
    else:
        raise ValueError(f"unsupported source baseline schema: {schema}")
    return migrated


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not args.overwrite:
        parser.error("--overwrite is required for an explicit migration")
    original = json.loads(args.path.read_text(encoding="utf-8"))
    migrated = migrate(original)
    args.path.write_text(
        json.dumps(
            migrated,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
