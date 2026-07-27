from __future__ import annotations

import ast
from pathlib import Path

from pol.paper1.e2_convergence import decide_convergence
from pol.paper1.e2_evaluation import evaluate_test
from pol.paper1.e2_points import SelectionDatasetView


ROOT = Path(__file__).resolve().parents[1]


def test_selection_module_has_no_test_dataset_dependency() -> None:
    source = (ROOT / "pol/paper1/e2_selection.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    assert "TestDatasetView" not in source
    assert "test_indices" not in source
    assert all(
        not (
            isinstance(node, ast.ImportFrom)
            and node.module == "pol.paper1.datasets"
        )
        for node in ast.walk(tree)
    )


def test_selection_view_has_no_test_fields() -> None:
    assert all(
        "test" not in name
        for name in SelectionDatasetView.__dataclass_fields__
    )


def test_rerun_decision_is_explicit_and_bounded() -> None:
    assert decide_convergence(
        pilot_n_sur=32, selected_base=64, reruns_remaining=1
    ).status == "rerun"
    assert decide_convergence(
        pilot_n_sur=32, selected_base=64, reruns_remaining=0
    ).status == "reject"


def test_run_e2_has_no_recursive_self_call() -> None:
    tree = ast.parse((ROOT / "pol/paper1/e2.py").read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "run_e2"
    )
    assert not any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "run_e2"
        for node in ast.walk(function)
    )


def test_production_attempt_calls_responsibility_module_apis() -> None:
    tree = ast.parse((ROOT / "pol/paper1/e2.py").read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_run_e2_attempt"
    )
    calls = {
        node.func.id
        for node in ast.walk(function)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert {
        "evaluate_validation_points",
        "select_validation_coordinates",
        "build_selection_result",
        "evaluate_convergence",
        "publish_and_read_back_frozen_plan",
        "evaluate_test",
    } <= calls


def test_selection_and_convergence_modules_have_no_test_capability() -> None:
    for name in ("e2_selection.py", "e2_convergence.py"):
        source = (ROOT / "pol/paper1" / name).read_text(encoding="utf-8")
        assert "TestDatasetView" not in source
        assert "Paper1MasterDataset" not in source
        assert "y_target_master" not in source


def test_convergence_callback_cannot_capture_full_dataset() -> None:
    """The production closure must narrow dataset authority to sample IDs."""
    tree = ast.parse((ROOT / "pol/paper1/e2.py").read_text(encoding="utf-8"))
    attempt = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_run_e2_attempt"
    )
    convergence_call = next(
        node
        for node in ast.walk(attempt)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "evaluate_convergence"
    )
    callback = next(
        keyword.value
        for keyword in convergence_call.keywords
        if keyword.arg == "evaluate"
    )
    assert isinstance(callback, ast.Lambda)
    assert not any(
        isinstance(node, ast.Name) and node.id == "dataset"
        for node in ast.walk(callback)
    )
    assert not any(
        argument.arg == "dataset"
        for argument in next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "_convergence"
        ).args.args
    )


def test_test_evaluation_requires_durable_frozen_reference() -> None:
    with __import__("pytest").raises(
        TypeError, match="FrozenPlanReference"
    ):
        from pol.paper1.e2_evaluation import TestEvaluationContext

        evaluate_test(context=TestEvaluationContext(
            config=None,
            frozen=object(),
            test_view=object(),
            evaluator=None,
            point_order=[],
            point_models={},
            point_selections={},
            pilot_n_sur=1,
            selection_record_hash="x",
            event_log=[],
            solve_state=lambda *args: None,
            build_features=lambda *args: None,
            metric_row=lambda *args: None,
        ))
