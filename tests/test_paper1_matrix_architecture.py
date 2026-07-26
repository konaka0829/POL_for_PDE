from __future__ import annotations

import ast
from pathlib import Path
import subprocess
import sys


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


def test_matrix_plan_does_not_import_compute_or_plotting_modules() -> None:
    code = """
from pathlib import Path
import sys
from pol.workflow.matrix_spec import load_matrix_spec
from pol.workflow.matrix import matrix_plan_to_dict
root = Path.cwd()
spec = load_matrix_spec(
    root / "configs/runs/paper1_e1_resolution_sweep_smoke.json",
    repo_root=root,
)
matrix_plan_to_dict(spec, repo_root=root)
assert "pol.paper1.e1" not in sys.modules
assert not any(name.startswith("matplotlib") for name in sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
