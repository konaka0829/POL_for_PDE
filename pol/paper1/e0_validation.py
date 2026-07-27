"""Recomputable, cross-artifact validation for complete Paper 1 E0 runs.

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


def validate_e0_scientific_artifacts(
    output_dir: Path,
    *,
    expected_config_identity: str | None = None,
    expected_config_path: Path | None = None,
) -> E0ScientificValidation:
    """Recompute every E0 gate record from its config and primitive archive."""
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
