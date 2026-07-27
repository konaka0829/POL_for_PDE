from __future__ import annotations

import ast
from pathlib import Path

from pol.paper1.e2_convergence import decide_convergence
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
