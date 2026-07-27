"""Artifact-only and opt-in deep validation for complete Paper 1 E0 runs.

The byte manifest detects accidental corruption and partial/stale publication.
This module additionally derives the scientific records from the saved
configuration and primitive master archive.  Without an external trust anchor
it cannot authenticate an attacker-controlled, fully regenerated experiment;
it does reject metadata-only forgeries and cross-artifact inconsistencies.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, replace
import json
import math
from pathlib import Path
from typing import Any, Mapping

from pol.runtime.hashing import stable_object_hash

from .config import canonical_config_json, load_config_json
from .datasets import tensor_hash
from .e0 import (
    E0SolverCache,
    build_required_checks,
    load_master_initial_conditions,
    run_algebraic_checks,
    run_interface_checks,
    run_model1_checks,
    run_reference_convergence,
)
from .protocols import E0_SCHEMA_VERSION


@dataclass(frozen=True)
class E0ScientificValidation:
    selected_reference: Mapping[str, object]
    accepted_config_identity: str
    master_tensor_hash: str
    scientific_identity: str
    canonical_records: Mapping[str, object]


def e0_scientific_identity_from_artifacts(output_dir: Path) -> str:
    """Rebuild the identity after a caller has validated the artifact set."""
    root = Path(output_dir)
    accepted = load_config_json(root / "accepted_production_config.json")
    summary = _load_json(root / "e0_summary.json")
    master_manifest = _load_json(root / "master_manifest.json")
    records = {
        name: _load_json(root / name)
        for name in (
            "resampling_checks.json",
            "reference_convergence.json",
            "input_interface_checks.json",
            "model1_identity.json",
        )
    }
    return stable_object_hash(
        {
            "protocol_version": E0_SCHEMA_VERSION,
            "accepted_config": canonical_config_json(accepted),
            "selected_reference": summary["selected_reference"],
            "master_tensor_hash": master_manifest["tensor_hash"],
            "checks": records,
        }
    )


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"E0 artifact must contain a JSON object: {path.name}")
    return value


def _require_equal(name: str, saved: object, expected: object) -> None:
    if saved != expected:
        raise ValueError(f"E0 scientific artifact mismatch: {name}")


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError("E0 reference CSV has missing/duplicate columns")
        return list(reader)


def _reference_csv_rows(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    def cell(value: object) -> str:
        return "" if value is None else str(value)

    result: list[dict[str, str]] = []
    for row in rows:
        result.append(
            {
                "kind": cell(row["kind"]),
                "candidate_nx": cell(row["candidate_nx"]),
                "eligible_for_production": cell(
                    row.get("eligible_for_production", "")
                ),
                "status": cell(row.get("status", "")),
                "solver": cell(row["solver"]),
                "requested_dt": cell(row["requested_dt"]),
                "requested_fine_dt": cell(row["requested_fine_dt"]),
                "outer_steps": cell(row["outer_steps"]),
                "substeps_per_outer": cell(row["substeps_per_outer"]),
                "effective_inner_step": cell(row["effective_inner_step"]),
                "relative_l2_mean": cell(row["relative_l2"]["mean"]),
                "relative_l2_median": cell(row["relative_l2"]["median"]),
                "relative_l2_max": cell(row["relative_l2"]["max"]),
                "absolute_l2_mean": cell(row["absolute_l2"]["mean"]),
                "low_mode_relative_l2_mean": cell(
                    row["low_mode_relative_l2"]["mean"]
                ),
                "master_hash": cell(row["master_hash"]),
                "sample_ids": json.dumps(row["sample_ids"]),
            }
        )
    return result


def deep_validate_e0_scientific_artifacts(
    output_dir: Path,
    *,
    expected_config_identity: str | None = None,
    expected_config_path: Path | None = None,
) -> E0ScientificValidation:
    """Recompute every E0 gate record; this is never used by normal reuse."""
    root = Path(output_dir)
    resolved = load_config_json(root / "resolved_config.json")
    if resolved.e0 is None:
        raise ValueError("E0 resolved config has no e0 section")
    if expected_config_identity is not None:
        actual = stable_object_hash(canonical_config_json(resolved))
        if actual != expected_config_identity:
            raise ValueError("E0 resolved config does not match current request")
    if expected_config_path is not None and expected_config_identity is None:
        requested = load_config_json(expected_config_path)
        if canonical_config_json(requested) != canonical_config_json(resolved):
            raise ValueError("E0 resolved config does not match current request")

    master = load_master_initial_conditions(
        root / "master_initial_conditions.pt", resolved
    )
    master_hash = tensor_hash(master.values_master)
    cache = E0SolverCache()
    expected_resampling, projector = run_algebraic_checks(resolved)
    expected_resampling["fourier_projector"] = projector
    expected_convergence = run_reference_convergence(resolved, master, cache=cache)
    reference_state = expected_convergence.pop("_reference_state")
    expected_interfaces = run_interface_checks(resolved, master, reference_state)
    expected_model1 = run_model1_checks(resolved, master, cache=cache)
    expected_checks = build_required_checks(
        expected_resampling,
        projector,
        expected_convergence,
        expected_interfaces,
        expected_model1,
    )

    saved_records = {
        "resampling_checks.json": _load_json(root / "resampling_checks.json"),
        "reference_convergence.json": _load_json(
            root / "reference_convergence.json"
        ),
        "input_interface_checks.json": _load_json(
            root / "input_interface_checks.json"
        ),
        "model1_identity.json": _load_json(root / "model1_identity.json"),
    }
    expected_records = {
        "resampling_checks.json": expected_resampling,
        "reference_convergence.json": expected_convergence,
        "input_interface_checks.json": expected_interfaces,
        "model1_identity.json": expected_model1,
    }
    for name, expected in expected_records.items():
        _require_equal(name, saved_records[name], expected)
    _require_equal(
        "reference_convergence.csv",
        _csv_rows(root / "reference_convergence.csv"),
        _reference_csv_rows(expected_convergence["rows"]),
    )

    chosen_space = expected_convergence.get("selected_spatial")
    chosen_time = expected_convergence.get("selected_temporal")
    if (
        not isinstance(chosen_space, dict)
        or not isinstance(chosen_time, dict)
        or expected_convergence.get("joint_status") != "pass"
    ):
        raise ValueError("E0 recomputed reference selection is not passing")
    selected = {
        "reference_nx": chosen_space["candidate_nx"],
        "solver": chosen_time["solver"],
        "requested_dt": chosen_time["requested_dt"],
        "requested_fine_dt": chosen_time["requested_fine_dt"],
        "effective_inner_step": chosen_time["effective_inner_step"],
        "joint_status": "pass",
    }
    expected_summary = {
        "schema_version": E0_SCHEMA_VERSION,
        "status": "pass",
        "required_checks": expected_checks,
        "selected_reference": selected,
        "accepted_production_config": "accepted_production_config.json",
        "num_failures": 0,
        "failure_reasons": [],
    }
    saved_summary = _load_json(root / "e0_summary.json")
    if saved_summary.get("selected_reference") != selected:
        raise ValueError("E0 summary/reference selection mismatch")
    _require_equal("e0_summary.json", saved_summary, expected_summary)

    accepted = load_config_json(root / "accepted_production_config.json")
    authorized = replace(
        resolved,
        e0=None,
        spatial=replace(
            resolved.spatial, reference_nx=int(selected["reference_nx"])
        ),
        target=replace(
            resolved.target,
            solver=str(selected["solver"]),
            dt=float(selected["requested_dt"]),
            fine_dt=selected["requested_fine_dt"],
        ),
    )
    if canonical_config_json(accepted) != canonical_config_json(authorized):
        raise ValueError("E0 accepted config contains unauthorized changes")
    accepted_identity = stable_object_hash(canonical_config_json(accepted))
    canonical_records = {
        "protocol_version": E0_SCHEMA_VERSION,
        "accepted_config": canonical_config_json(accepted),
        "selected_reference": selected,
        "master_tensor_hash": master_hash,
        "checks": expected_records,
    }
    return E0ScientificValidation(
        selected_reference=selected,
        accepted_config_identity=accepted_identity,
        master_tensor_hash=master_hash,
        scientific_identity=stable_object_hash(canonical_records),
        canonical_records=canonical_records,
    )


def _require_pass_statuses(value: object, path: str = "$") -> None:
    """Require every persisted E0 status and boolean check to be passing."""
    if isinstance(value, dict):
        if "status" in value and value["status"] != "pass":
            raise ValueError(f"E0 scientific artifact mismatch: {path}.status")
        if "checks" in value and isinstance(value["checks"], dict):
            for name, check in value["checks"].items():
                if isinstance(check, bool) and not check:
                    raise ValueError(
                        f"E0 scientific artifact mismatch: {path}.checks.{name}"
                    )
        for key, item in value.items():
            _require_pass_statuses(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _require_pass_statuses(item, f"{path}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"E0 scientific artifact mismatch: non-finite {path}")


def validate_e0_scientific_artifacts(
    output_dir: Path,
    *,
    expected_config_identity: str | None = None,
    expected_config_path: Path | None = None,
) -> E0ScientificValidation:
    """Validate E0 solely from its persisted, byte-verified artifacts."""
    root = Path(output_dir)
    resolved = load_config_json(root / "resolved_config.json")
    if resolved.e0 is None:
        raise ValueError("E0 resolved config has no e0 section")
    if expected_config_identity is not None:
        actual = stable_object_hash(canonical_config_json(resolved))
        if actual != expected_config_identity:
            raise ValueError("E0 resolved config does not match current request")
    if expected_config_path is not None and expected_config_identity is None:
        requested = load_config_json(expected_config_path)
        if canonical_config_json(requested) != canonical_config_json(resolved):
            raise ValueError("E0 resolved config does not match current request")

    records = {
        "resampling_checks.json": _load_json(root / "resampling_checks.json"),
        "reference_convergence.json": _load_json(
            root / "reference_convergence.json"
        ),
        "input_interface_checks.json": _load_json(
            root / "input_interface_checks.json"
        ),
        "model1_identity.json": _load_json(root / "model1_identity.json"),
    }
    for name, record in records.items():
        if record.get("schema_version") != E0_SCHEMA_VERSION:
            raise ValueError(f"E0 scientific artifact mismatch: {name}.schema_version")
        _require_pass_statuses(record, f"$.{name}")

    resampling = records["resampling_checks.json"]
    projector = resampling.get("fourier_projector")
    if not isinstance(projector, dict):
        raise ValueError("E0 scientific artifact mismatch: fourier_projector")
    for check in resampling.get("checks", {}).values():
        if not isinstance(check, dict):
            raise ValueError("E0 scientific artifact mismatch: resampling check")
        error = check.get("max_abs_error")
        if isinstance(error, (int, float)):
            atol = float(check.get("atol", 1e-6))
            rtol = float(check.get("rtol", 0.0))
            if not math.isfinite(float(error)) or float(error) > atol + 2.0 * rtol:
                raise ValueError(
                    "E0 scientific artifact mismatch: resampling threshold"
                )
    convergence = records["reference_convergence.json"]
    interfaces = records["input_interface_checks.json"]
    model1 = records["model1_identity.json"]
    required_checks = build_required_checks(
        resampling, projector, convergence, interfaces, model1
    )
    if any(status != "pass" for status in required_checks.values()):
        raise ValueError("E0 scientific artifact mismatch: required checks")
    algebraic_limit = (
        resolved.e0.algebraic_tolerances.float64_atol
        + 2.0 * resolved.e0.algebraic_tolerances.float64_rtol
    )
    if (
        float(interfaces["no_high_frequency_leak"]["max_finite_difference"])
        > algebraic_limit
        or float(
            interfaces["no_high_frequency_leak"]["max_surrogate_difference"]
        )
        > algebraic_limit
        or float(
            interfaces["target_coefficient_consistency"]["max_abs_error"]
        )
        > algebraic_limit
        or float(model1["bandlimited_reduced_j"]["max_abs_error"])
        > algebraic_limit
        or any(
            float(model1["full_observation"][name]) > algebraic_limit
            for name in (
                "terminal_max_abs_error",
                "coefficient_max_abs_error",
                "projection_max_abs_error",
            )
        )
        or float(model1["aliasing_counterexample"]["max_abs_difference"])
        <= algebraic_limit
    ):
        raise ValueError("E0 scientific artifact mismatch: metric threshold")

    rows = convergence.get("rows")
    if not isinstance(rows, list):
        raise ValueError("E0 scientific artifact mismatch: convergence rows")
    _require_equal(
        "reference_convergence.csv",
        _csv_rows(root / "reference_convergence.csv"),
        _reference_csv_rows(rows),
    )
    tolerances = convergence.get("tolerances")
    if not isinstance(tolerances, dict):
        raise ValueError("E0 scientific artifact mismatch: convergence tolerances")
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("E0 scientific artifact mismatch: convergence row")
        if row.get("status") == "pass" or row in (
            convergence.get("selected_spatial"),
            convergence.get("selected_temporal"),
            convergence.get("joint_row"),
        ):
            if (
                float(row["relative_l2"]["mean"])
                > float(tolerances["mean_relative_l2"])
                or float(row["relative_l2"]["max"])
                > float(tolerances["max_relative_l2"])
                or float(row["low_mode_relative_l2"]["mean"])
                > float(tolerances["low_mode_relative_l2"])
            ):
                raise ValueError(
                    "E0 scientific artifact mismatch: convergence threshold"
                )

    chosen_space = convergence.get("selected_spatial")
    chosen_time = convergence.get("selected_temporal")
    joint = convergence.get("joint_row")
    if not all(isinstance(item, dict) for item in (chosen_space, chosen_time, joint)):
        raise ValueError("E0 scientific artifact mismatch: reference selection")
    eligible_spatial = [
        row for row in rows
        if row.get("kind") == "spatial"
        and row.get("eligible_for_production") is True
        and float(row["relative_l2"]["mean"])
        <= float(tolerances["mean_relative_l2"])
        and float(row["relative_l2"]["max"])
        <= float(tolerances["max_relative_l2"])
        and float(row["low_mode_relative_l2"]["mean"])
        <= float(tolerances["low_mode_relative_l2"])
    ]
    if not eligible_spatial or chosen_space != min(
        eligible_spatial, key=lambda row: int(row["candidate_nx"])
    ):
        raise ValueError("E0 scientific artifact mismatch: selection rule")
    selected = {
        "reference_nx": chosen_space["candidate_nx"],
        "solver": chosen_time["solver"],
        "requested_dt": chosen_time["requested_dt"],
        "requested_fine_dt": chosen_time["requested_fine_dt"],
        "effective_inner_step": chosen_time["effective_inner_step"],
        "joint_status": "pass",
    }
    summary = _load_json(root / "e0_summary.json")
    if summary.get("selected_reference") != selected:
        raise ValueError("E0 summary/reference selection mismatch")
    expected_summary = {
        "schema_version": E0_SCHEMA_VERSION,
        "status": "pass",
        "required_checks": required_checks,
        "selected_reference": selected,
        "accepted_production_config": "accepted_production_config.json",
        "num_failures": 0,
        "failure_reasons": [],
    }
    _require_equal("e0_summary.json", summary, expected_summary)

    accepted = load_config_json(root / "accepted_production_config.json")
    authorized = replace(
        resolved,
        e0=None,
        spatial=replace(
            resolved.spatial, reference_nx=int(selected["reference_nx"])
        ),
        target=replace(
            resolved.target,
            solver=str(selected["solver"]),
            dt=float(selected["requested_dt"]),
            fine_dt=selected["requested_fine_dt"],
        ),
    )
    if canonical_config_json(accepted) != canonical_config_json(authorized):
        raise ValueError("E0 accepted config contains unauthorized changes")
    master = load_master_initial_conditions(
        root / "master_initial_conditions.pt", accepted
    )
    master_hash = tensor_hash(master.values_master)
    manifest = _load_json(root / "master_manifest.json")
    if manifest.get("tensor_hash") != master_hash:
        raise ValueError("E0 master manifest tensor hash mismatch")
    canonical_records = {
        "protocol_version": E0_SCHEMA_VERSION,
        "accepted_config": canonical_config_json(accepted),
        "selected_reference": selected,
        "master_tensor_hash": master_hash,
        "checks": records,
    }
    return E0ScientificValidation(
        selected_reference=selected,
        accepted_config_identity=stable_object_hash(
            canonical_config_json(accepted)
        ),
        master_tensor_hash=master_hash,
        scientific_identity=stable_object_hash(canonical_records),
        canonical_records=canonical_records,
    )
