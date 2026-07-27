from __future__ import annotations

import ast
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
ACTIVE_ROOTS = (
    ROOT / "pol/cli.py",
    ROOT / "pol/paper1",
    ROOT / "pol/runtime",
    ROOT / "pol/workflow",
    ROOT / "pol/plots",
)
LEGACY_PREFIXES = (
    "pol.model123_1d",
    "pol.burgers_spectral_1d",
    "pol.spectral_etdrk4_1d",
    "pol.reservoir_1d",
    "pol.elm",
)


def _active_python_files() -> list[Path]:
    files = []
    for root in ACTIVE_ROOTS:
        files.extend([root] if root.is_file() else root.rglob("*.py"))
    return files


def test_active_import_graph_has_no_legacy_edge() -> None:
    for path in _active_python_files():
        imports = []
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.append(node.module)
        assert not any(
            name.startswith(LEGACY_PREFIXES) for name in imports
        ), (path, imports)


def test_legacy_paths_are_absent_from_active_tree() -> None:
    for relative in (
        "pol/model123_1d",
        "pol/burgers_spectral_1d.py",
        "pol/spectral_etdrk4_1d.py",
        "model123_burgers_1d.py",
        "model123_error_study.py",
        "model1_error_decomposition_1d.py",
        "configs/B0_smoke.json",
        "configs/B1_burgers_grf.json",
        "legacy",
    ):
        assert not (ROOT / relative).exists(), relative


def test_import_pol_is_minimal() -> None:
    code = """
import pol
import sys
assert not any(name.startswith(('pol.model123_1d',
 'pol.burgers_spectral_1d', 'pol.spectral_etdrk4_1d'))
 for name in sys.modules)
for name in ('FixedRandomELM', 'DatasetConfig', 'Reservoir1DSolver'):
    assert not hasattr(pol, name)
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_archive_tag_points_to_documented_commit() -> None:
    expected = "9320d1e86c7a2212f56cf4c64a326727cdd03b43"
    result = subprocess.run(
        ["git", "rev-list", "-n", "1", "pre-phase6-model123"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0
    assert result.stdout.strip() == expected
    restored = subprocess.run(
        [
            "git",
            "show",
            "pre-phase6-model123:pol/model123_1d/experiments.py",
        ],
        cwd=ROOT,
        capture_output=True,
        check=False,
    )
    assert restored.returncode == 0
