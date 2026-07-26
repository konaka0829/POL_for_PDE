from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_generic_matrix_engine_has_no_e1_table_or_thread_pool_logic() -> None:
    sources = [
        ROOT / "pol/workflow/matrix.py",
        ROOT / "pol/workflow/matrix_spec.py",
        ROOT / "pol/workflow/matrix_worker.py",
    ]
    combined = "\n".join(path.read_text(encoding="utf-8") for path in sources)
    for scientific_column in (
        "selected_results",
        "readout_diagnostics",
        "noise_summary",
        "target_data_nx",
        "observation_dim",
    ):
        assert scientific_column not in combined
    assert "ThreadPoolExecutor" not in combined
    assert "ProcessPoolExecutor" in (ROOT / "pol/workflow/matrix.py").read_text()


def test_generic_matrix_does_not_import_e1_scientific_modules() -> None:
    for path in (ROOT / "pol/workflow").glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.append(node.module)
        assert not any(
            name.startswith(("pol.paper1.e1", "pol.paper1.readouts"))
            for name in imports
        ), (path, imports)
