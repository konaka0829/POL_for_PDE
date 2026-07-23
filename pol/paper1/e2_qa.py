"""Read-back and artifact QA helpers for Paper 1 E2."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch


def file_record(path: Path, root: Path) -> dict[str, Any]:
    return {"relative_path": str(path.relative_to(root)), "size_bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def assert_finite(value: Any, path: str = "root") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"non-finite value at {path}")
    if isinstance(value, dict):
        for key, item in value.items(): assert_finite(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value): assert_finite(item, f"{path}[{index}]")


def validate_csv(path: Path, required: set[str], unique: tuple[str, ...]) -> int:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or not required <= set(rows[0]):
        raise ValueError(f"{path.name} missing rows/columns: {sorted(required-set(rows[0] if rows else []))}")
    keys = [tuple(row[key] for key in unique) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError(f"{path.name} contains duplicate keys")
    for row in rows:
        for key, value in row.items():
            if value and key not in {"family", "sweep_axis", "model", "status", "frozen_readout",
                                     "state_cache_key", "feature_cache_key", "selection_record_hash"}:
                try:
                    number = float(value)
                except ValueError:
                    continue
                if not math.isfinite(number):
                    raise ValueError(f"{path.name} has non-finite {key}")
    return len(rows)


def write_manifest(output_dir: Path, expected: set[str]) -> None:
    records = [file_record(output_dir / name, output_dir) for name in sorted(expected)]
    path = output_dir / "artifact_manifest.json"
    path.write_text(json.dumps({"schema_version": "paper1-e2-artifacts-v1", "files": records},
                               indent=2, sort_keys=True, allow_nan=False) + "\n")
    loaded = json.loads(path.read_text())
    for record in loaded["files"]:
        target = output_dir / record["relative_path"]
        if target.stat().st_size != record["size_bytes"] or hashlib.sha256(target.read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError("artifact manifest verification failed")


def validate_resume_output(output_dir: Path) -> bool:
    summary = json.loads((output_dir / "e2_summary.json").read_text())
    manifest = json.loads((output_dir / "artifact_manifest.json").read_text())
    if summary.get("status") != "pass":
        return False
    recorded = {record["relative_path"] for record in manifest["files"]}
    actual = {path.name for path in output_dir.iterdir() if path.is_file() and path.name != "artifact_manifest.json"}
    if recorded != actual:
        raise ValueError(f"resume artifact set mismatch: missing={sorted(recorded-actual)}, extra={sorted(actual-recorded)}")
    for record in manifest["files"]:
        path = output_dir / record["relative_path"]
        if not path.exists() or path.stat().st_size != record["size_bytes"] or hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError(f"resume artifact integrity check failed: {path.name}")
    return True
