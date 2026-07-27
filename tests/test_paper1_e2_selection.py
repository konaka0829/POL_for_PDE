from types import SimpleNamespace

import pytest
import torch

from pol.paper1.config import load_config_json
from pol.paper1.datasets import _split_indices
from pol.paper1.e2 import select_first_with_tolerance, validate_convergence_membership
from pol.paper1.e2_selection import (
    assert_validation_only_record,
    build_selection_bindings,
)


def test_validation_selection_uses_config_order_and_not_test():
    rows=[
        {"validation_field_relative_l2_mean":.2,"test":.01,"parameter":1},
        {"validation_field_relative_l2_mean":.1,"test":.9,"parameter":2},
        {"validation_field_relative_l2_mean":.1000001,"test":.001,"parameter":3},
    ]
    assert select_first_with_tolerance(rows,"validation_field_relative_l2_mean",1e-8)["parameter"]==2
    assert "test" not in select_first_with_tolerance.__code__.co_varnames


def test_actual_shuffled_membership_rejects_small_test_id():
    config = load_config_json("configs/paper1_e2_main.json")
    train, validation, test, _ = _split_indices(config)
    dataset = SimpleNamespace(
        sample_ids=torch.arange(config.data.total_samples),
        train_indices=train, val_indices=validation, test_indices=test)
    assert 2 in test.tolist()
    with pytest.raises(ValueError, match="test sample ID 2"):
        validate_convergence_membership(dataset, (0, 2))
    membership = validate_convergence_membership(
        dataset, config.e2.convergence.sample_ids)
    assert set(membership.values()) == {"validation"}
def test_selection_binding_is_independent_of_test_labels() -> None:
    train = torch.tensor([0, 1])
    validation = torch.tensor([2])
    u0 = torch.arange(12, dtype=torch.float64).reshape(4, 3)
    coefficients = u0.clone()
    target_data = u0.clone()
    reference = u0.clone()
    first = build_selection_bindings(
        {"dataset_hash": "full", "split_hash": "split"},
        train_indices=train,
        validation_indices=validation,
        u0_data=u0,
        target_coefficients=coefficients,
        target_data=target_data,
        reference=reference,
    )
    coefficients[3] += 1000
    target_data[3] += 1000
    reference[3] += 1000
    second = build_selection_bindings(
        {"dataset_hash": "changed", "split_hash": "split"},
        train_indices=train,
        validation_indices=validation,
        u0_data=u0,
        target_coefficients=coefficients,
        target_data=target_data,
        reference=reference,
    )
    assert first == second
    assert "dataset_hash" not in first


def test_nested_test_binding_is_rejected_with_path() -> None:
    with pytest.raises(
        ValueError, match=r"\$\.bindings\.nested\[0\]\.test_target_hash"
    ):
        assert_validation_only_record(
            {"bindings": {"nested": [{"test_target_hash": "forbidden"}]}}
        )
    with pytest.raises(
        ValueError, match=r"\$\.bindings\.nested\.dataset_hash"
    ):
        assert_validation_only_record(
            {"bindings": {"nested": {"dataset_hash": "full-dataset"}}}
        )
