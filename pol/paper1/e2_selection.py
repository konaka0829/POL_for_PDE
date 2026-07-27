"""Validation-only E2 selection result contracts."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch

from .e2_cache import tensor_hash
from .e2_cache import stable_hash

@dataclass(frozen=True)
class SelectionResult:
    """Selections and durable record produced without test tensors."""

    representatives: dict[str, dict[str, Any]]
    model_specific_optima: dict[str, Any]
    selection_record: dict[str, Any]
    selection_record_hash: str


def build_selection_result(
    *,
    protocol_version: str,
    bindings: dict[str, Any],
    representatives: dict[str, dict[str, Any]],
    model_specific_optima: dict[str, Any],
    point_hyperparameters: dict[str, Any],
) -> SelectionResult:
    """Construct and hash the validation-only durable selection record."""
    record = {
        "schema_version": protocol_version,
        "protocol_version": protocol_version,
        "bindings": bindings,
        "representatives": representatives,
        "model_specific_optima": model_specific_optima,
        "point_hyperparameters": point_hyperparameters,
        "test_data_used": False,
    }
    assert_validation_only_record(record)
    return SelectionResult(
        representatives=representatives,
        model_specific_optima=model_specific_optima,
        selection_record=record,
        selection_record_hash=stable_hash(record),
    )


def select_validation_coordinates(
    *,
    e2_config: Any,
    validation_rows: list[dict[str, Any]],
    evaluate_point: Callable[[str, str, float, float], None],
    models: tuple[str, ...],
) -> tuple[
    dict[str, Any],
    dict[str, dict[str, Any]],
    dict[str, tuple[str, str, float, float]],
]:
    """Run validation-only coordinate selection in configured candidate order."""
    model_specific: dict[str, Any] = {}
    representatives: dict[str, dict[str, Any]] = {}
    representative_identities: dict[
        str, tuple[str, str, float, float]
    ] = {}
    for family, family_cfg in (
        ("burgers", e2_config.burgers),
        ("reaction_diffusion", e2_config.reaction_diffusion),
    ):
        for nu in family_cfg.nu_grid:
            evaluate_point(
                family, "nu_tilde", nu, family_cfg.initial_T_anchor
            )
        model_specific[family] = {}
        for model in models:
            nu_rows = [
                row
                for row in validation_rows
                if row["family"] == family
                and row["sweep_axis"] == "nu_tilde"
                and row["model"] == model
            ]
            nu_star = float(
                select_first_with_tolerance(
                    nu_rows,
                    e2_config.selection_metric,
                    e2_config.parameter_tie_tolerance,
                )["nu_tilde"]
            )
            time_axis = (
                "T_tilde"
                if model == e2_config.representative_model
                else f"T_tilde_{model}"
            )
            for T in family_cfg.T_grid:
                evaluate_point(family, time_axis, nu_star, T)
            time_rows = [
                row
                for row in validation_rows
                if row["family"] == family
                and row["sweep_axis"] == time_axis
                and row["model"] == model
            ]
            selected = select_first_with_tolerance(
                time_rows,
                e2_config.selection_metric,
                e2_config.parameter_tie_tolerance,
            )
            T_star = float(selected["T_tilde"])
            for round_index in range(e2_config.coordinate_refinement_rounds):
                suffix = round_index + 1
                nu_axis = (
                    f"nu_refinement_{suffix}"
                    if model == e2_config.representative_model
                    else f"nu_refinement_{suffix}_{model}"
                )
                for nu in family_cfg.nu_grid:
                    evaluate_point(family, nu_axis, nu, T_star)
                nu_star = float(
                    select_first_with_tolerance(
                        [
                            row
                            for row in validation_rows
                            if row["family"] == family
                            and row["sweep_axis"] == nu_axis
                            and row["model"] == model
                        ],
                        e2_config.selection_metric,
                        e2_config.parameter_tie_tolerance,
                    )["nu_tilde"]
                )
                time_axis = (
                    f"T_refinement_{suffix}"
                    if model == e2_config.representative_model
                    else f"T_refinement_{suffix}_{model}"
                )
                for T in family_cfg.T_grid:
                    evaluate_point(family, time_axis, nu_star, T)
                selected = select_first_with_tolerance(
                    [
                        row
                        for row in validation_rows
                        if row["family"] == family
                        and row["sweep_axis"] == time_axis
                        and row["model"] == model
                    ],
                    e2_config.selection_metric,
                    e2_config.parameter_tie_tolerance,
                )
                T_star = float(selected["T_tilde"])
            model_specific[family][model] = {
                "nu_tilde": nu_star,
                "T_tilde": T_star,
                "validation_field_relative_l2_mean": selected[
                    e2_config.selection_metric
                ],
                "coordinate_path": "independent_validation_only",
            }
            if model == e2_config.representative_model:
                representatives[family] = {
                    "nu_star": nu_star,
                    "T_star": T_star,
                    "representative_model": model,
                    "selection_metric": e2_config.selection_metric,
                }
                representative_identities[family] = (
                    family,
                    time_axis,
                    nu_star,
                    T_star,
                )
    return model_specific, representatives, representative_identities


def select_first_with_tolerance(
    rows: list[dict[str, Any]], metric: str, tolerance: float
) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot select from empty validation rows")
    best = min(float(row[metric]) for row in rows)
    return next(
        row for row in rows if float(row[metric]) <= best + tolerance
    )


def assert_validation_only_record(record: dict[str, Any]) -> None:
    """Fail closed if a selection record acquires test-label bindings."""
    forbidden = {
        "dataset_hash",
        "dataset_prerequisite_hash",
        "input_tensor_hashes",
        "finite_input_tensor_hashes",
        "test_target_hash",
        "test_labels",
        "test_target_coefficients",
        "test_reference_hash",
        "full_target_hash",
        "dataset_target_hash",
    }

    def scan(value: Any, path: str) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                normalized = str(key).lower()
                if normalized in forbidden:
                    raise ValueError(
                        "selection record contains forbidden full/test binding "
                        f"at {path}.{key}"
                    )
                scan(item, f"{path}.{key}")
        elif isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                scan(item, f"{path}[{index}]")

    scan(record.get("bindings", {}), "$.bindings")


def build_selection_bindings(
    base: dict[str, Any],
    *,
    train_indices: torch.Tensor,
    validation_indices: torch.Tensor,
    u0_data: torch.Tensor,
    target_coefficients: torch.Tensor,
    target_data: torch.Tensor,
    reference: torch.Tensor,
) -> dict[str, Any]:
    """Bind selection inputs without hashing any test-label/full-target tensor."""
    non_target = {
        key: value
        for key, value in base.items()
        if key
        not in {
            "dataset_hash",
            "dataset_prerequisite_hash",
            "input_tensor_hashes",
            "finite_input_tensor_hashes",
        }
    }
    return {
        **non_target,
        "split_identity": {
            "split_hash": base.get("split_hash"),
            "sample_ids_hash": base.get("sample_ids_hash"),
            "train_indices": tensor_hash(train_indices),
            "validation_indices": tensor_hash(validation_indices),
        },
        "selection_tensor_hashes": {
            "u0_data": tensor_hash(u0_data),
            "target_coefficients_train": tensor_hash(
                target_coefficients[train_indices]
            ),
            "target_coefficients_validation": tensor_hash(
                target_coefficients[validation_indices]
            ),
            "target_data_validation": tensor_hash(target_data[validation_indices]),
            "reference_validation": tensor_hash(reference[validation_indices]),
        },
    }
