"""Read-back and artifact QA helpers for Paper 1 E2."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch

from .e2 import E2_SCHEMA_VERSION, validate_frozen_evaluation_plan

ARTIFACT_MANIFEST_SCHEMA = "paper1-e2-artifacts-v3"
CSV_CONTRACT_VERSION = "paper1-e2-csv-v3"
ALLOWED_DIRECTORIES = {"cache"}
COMMON_FILES = {
    "resolved_config.json", "e0_prerequisite.json",
    "dataset_prerequisite.json", "data_manifest.json",
    "experiment_plan.json", "coordinate_history.json",
    "selection_record.json", "validation_sweep.csv",
    "model3_validation_by_seed.csv", "solver_metadata.csv",
    "physical_point_aliases.csv", "runtime_diagnostics.json",
    "model_specific_optima.json", "shared_representatives.json",
    "convergence_results.csv", "convergence_summary.json",
    "e2_attempt_history.json", "e2_summary.json", "failed_runs.json",
    "plot_manifest.json", "environment.json", "event_log.json",
    "artifact_manifest.json",
}
PASS_FILES = COMMON_FILES | {
    "frozen_evaluation_plan.pt", "test_sweep.csv",
    "model3_test_by_seed.csv", "model3_test_aggregate.csv",
    "selected_models.pt", "e2_handoff.json",
}
FAIL_PRETEST_FILES = COMMON_FILES
FAIL_RUNTIME_FILES = {
    "resolved_config.json", "e0_prerequisite.json",
    "dataset_prerequisite.json", "environment.json",
    "failed_runs.json", "e2_summary.json", "artifact_manifest.json",
}
PLOT_FILES = {
    "e2_parameter_sweeps.png", "e2_parameter_sweeps.pdf",
    "e2_nsur_convergence.png", "e2_nsur_convergence.pdf",
}
VERSIONED_CSV_FILES = {
    "validation_sweep.csv", "test_sweep.csv",
    "model3_validation_by_seed.csv", "model3_test_by_seed.csv",
    "model3_test_aggregate.csv", "convergence_results.csv",
    "solver_metadata.csv", "physical_point_aliases.csv",
}


def expected_artifacts(
        *, status: str, skip_plots: bool, test_evaluated: bool,
        failure_kind: str | None = None) -> set[str]:
    if status == "pass" and test_evaluated:
        return PASS_FILES | (set() if skip_plots else PLOT_FILES)
    if failure_kind == "runtime_error":
        return FAIL_RUNTIME_FILES
    return FAIL_PRETEST_FILES


def validate_artifact_contract(
        output_dir: Path, *, status: str, skip_plots: bool,
        test_evaluated: bool, include_manifest: bool = True,
        failure_kind: str | None = None) -> set[str]:
    expected = expected_artifacts(
        status=status, skip_plots=skip_plots,
        test_evaluated=test_evaluated, failure_kind=failure_kind)
    if not include_manifest:
        expected = expected - {"artifact_manifest.json"}
    actual_files = {p.name for p in output_dir.iterdir() if p.is_file()}
    actual_dirs = {p.name for p in output_dir.iterdir() if p.is_dir()}
    if actual_files != expected:
        raise ValueError(
            f"artifact contract mismatch: missing={sorted(expected-actual_files)}, "
            f"extra={sorted(actual_files-expected)}")
    if actual_dirs - ALLOWED_DIRECTORIES:
        raise ValueError(
            f"unknown artifact directories: {sorted(actual_dirs-ALLOWED_DIRECTORIES)}")
    return expected


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
    if (path.name in VERSIONED_CSV_FILES
            and any(row.get("schema_version") != CSV_CONTRACT_VERSION for row in rows)):
        raise ValueError(f"{path.name} CSV protocol mismatch")
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


def validate_result_cartesian(result: dict[str, Any], config: Any) -> None:
    """Validate exact row sets from the resolved in-memory experiment plan."""
    aliases = {
        (row["family"], row["sweep_axis"], float(row["nu_tilde"]),
         float(row["T_tilde"]))
        for row in result["physical_point_aliases"]}
    expected_validation = {
        (*alias, model) for alias in aliases
        for model in ("model1", "model2", "model3")}
    actual_validation = [
        (row["family"], row["sweep_axis"], float(row["nu_tilde"]),
         float(row["T_tilde"]), row["model"])
        for row in result["validation_sweep"]]
    if len(actual_validation) != len(set(actual_validation)):
        raise ValueError("duplicate validation Cartesian row")
    if set(actual_validation) != expected_validation:
        raise ValueError("validation Cartesian row mismatch")
    candidate_count = (
        len(config.e2.model3.widths)
        * len(config.e2.model3.weight_scales)
        * len(config.e2.model3.bias_scales)
        * len(config.e2.ridge.zetas))
    if len(result["model3_validation_by_seed"]) != (
            len(aliases) * candidate_count
            * len(config.e2.model3.selection_seeds)):
        raise ValueError("Model 3 validation Cartesian row count mismatch")
    if result["test_evaluated"]:
        actual_test = [
            (row["family"], row["sweep_axis"], float(row["nu_tilde"]),
             float(row["T_tilde"]), row["model"])
            for row in result["test_sweep"]]
        if len(actual_test) != len(set(actual_test)) or set(actual_test) != expected_validation:
            raise ValueError("test Cartesian row mismatch")
        if len(result["model3_test_by_seed"]) != (
                len(aliases) * len(config.e2.model3.evaluation_seeds)):
            raise ValueError("Model 3 test seed Cartesian row count mismatch")


def write_manifest(output_dir: Path, expected: set[str]) -> None:
    records = [file_record(output_dir / name, output_dir) for name in sorted(expected)]
    path = output_dir / "artifact_manifest.json"
    path.write_text(json.dumps({"schema_version": ARTIFACT_MANIFEST_SCHEMA, "files": records},
                               indent=2, sort_keys=True, allow_nan=False) + "\n")
    loaded = json.loads(path.read_text())
    for record in loaded["files"]:
        target = output_dir / record["relative_path"]
        if target.stat().st_size != record["size_bytes"] or hashlib.sha256(target.read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError("artifact manifest verification failed")


def validate_resume_output(output_dir: Path) -> bool:
    summary = json.loads((output_dir / "e2_summary.json").read_text())
    manifest = json.loads((output_dir / "artifact_manifest.json").read_text())
    if summary.get("schema_version") != E2_SCHEMA_VERSION:
        raise ValueError("resume E2 summary protocol mismatch")
    if manifest.get("schema_version") != ARTIFACT_MANIFEST_SCHEMA:
        raise ValueError("resume artifact manifest protocol mismatch")
    if summary.get("status") != "pass":
        return False
    selection = json.loads((output_dir / "selection_record.json").read_text())
    if selection.get("schema_version") != E2_SCHEMA_VERSION:
        raise ValueError("resume selection protocol mismatch")
    handoff = json.loads((output_dir / "e2_handoff.json").read_text())
    if handoff.get("schema_version") != "paper1-e2-handoff-v3":
        raise ValueError("resume handoff protocol mismatch")
    plot = json.loads((output_dir / "plot_manifest.json").read_text())
    if plot.get("schema_version") != "paper1-e2-plots-v3":
        raise ValueError("resume plot manifest protocol mismatch")
    plan = validate_frozen_evaluation_plan(
        output_dir / "frozen_evaluation_plan.pt",
        expected_selection_hash=summary.get("selection_record_hash"))
    if plan["plan_content_hash"] != summary.get("frozen_plan_hash"):
        raise ValueError("resume frozen plan hash mismatch")
    recorded = {record["relative_path"] for record in manifest["files"]}
    actual = {path.name for path in output_dir.iterdir() if path.is_file() and path.name != "artifact_manifest.json"}
    if recorded != actual:
        raise ValueError(f"resume artifact set mismatch: missing={sorted(recorded-actual)}, extra={sorted(actual-recorded)}")
    for record in manifest["files"]:
        path = output_dir / record["relative_path"]
        if not path.exists() or path.stat().st_size != record["size_bytes"] or hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError(f"resume artifact integrity check failed: {path.name}")
    dirs = {path.name for path in output_dir.iterdir() if path.is_dir()}
    if dirs - ALLOWED_DIRECTORIES:
        raise ValueError(
            f"resume artifact directory mismatch: extra={sorted(dirs-ALLOWED_DIRECTORIES)}")
    skip_plots = plot.get("status") == "skipped"
    expected = expected_artifacts(
        status="pass", skip_plots=skip_plots, test_evaluated=True)
    if recorded | {"artifact_manifest.json"} != expected:
        raise ValueError("resume explicit artifact contract mismatch")
    return True
