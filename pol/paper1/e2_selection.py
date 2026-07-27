"""Validation-only E2 selection result contracts."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .e2_cache import tensor_hash

@dataclass(frozen=True)
class SelectionResult:
    """Selections and durable record produced without test tensors."""

    representatives: dict[str, dict[str, Any]]
    model_specific_optima: dict[str, Any]
    selection_record: dict[str, Any]
    selection_record_hash: str


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
