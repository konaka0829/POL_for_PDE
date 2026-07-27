from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
RECIPE_MODULES = (
    "pol.paper1.recipes.foundation_validation",
    "pol.paper1.recipes.master_dataset",
    "pol.paper1.recipes.heat_calibration",
    "pol.paper1.recipes.surrogate_parameter_time",
)


@pytest.mark.parametrize("module_name", RECIPE_MODULES)
def test_recipe_module_import_is_side_effect_free(
    tmp_path: Path, module_name: str
) -> None:
    """Recipe imports must not parse arguments, print, or create files."""
    code = (
        "import importlib\n"
        "import sys\n"
        f"importlib.import_module({module_name!r})\n"
        "for name in ('pol.model123_1d', 'pol.reservoir_1d', 'pol.elm'):\n"
        "    assert name not in sys.modules, name\n"
    )
    before = set(tmp_path.iterdir())
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        env={"PYTHONPATH": str(ROOT)},
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == ""
    assert result.stderr == ""
    assert set(tmp_path.iterdir()) == before


@pytest.mark.parametrize("recipe", ("e1", "e2"))
def test_recursive_output_recipe_rejects_empty_symlink(
    tmp_path: Path, recipe: str
) -> None:
    target = tmp_path / "target"
    target.mkdir()
    output = tmp_path / "output"
    try:
        output.symlink_to(target, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink creation unavailable: {exc}")
    arguments = [
        sys.executable,
        str(ROOT / "tests/paper1_recipe_driver.py"),
        recipe,
        "--config",
        str(ROOT / (
            "configs/paper1_e1_smoke.json"
            if recipe == "e1"
            else "configs/paper1_e2_smoke.json"
        )),
        "--e0-dir",
        str(tmp_path / "missing-e0"),
        "--output-dir",
        str(output),
    ]
    if recipe == "e2":
        arguments.extend(["--dataset-dir", str(tmp_path / "missing-dataset")])
    result = subprocess.run(
        arguments,
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert result.returncode == 2
    assert "must not be a symlink" in result.stderr
    assert list(target.iterdir()) == []


@pytest.mark.parametrize("recipe", ("e0", "dataset", "e1", "e2"))
def test_missing_config_is_usage_error_without_output(
    tmp_path: Path, recipe: str
) -> None:
    output = tmp_path / "output"
    arguments = [
        sys.executable,
        str(ROOT / "tests/paper1_recipe_driver.py"),
        recipe,
        "--config",
        str(tmp_path / "missing.json"),
        "--output-dir",
        str(output),
    ]
    if recipe == "e1":
        arguments.extend(["--e0-dir", str(tmp_path / "missing-e0")])
    elif recipe == "e2":
        arguments.extend([
            "--e0-dir", str(tmp_path / "missing-e0"),
            "--dataset-dir", str(tmp_path / "missing-dataset"),
        ])
    result = subprocess.run(
        arguments,
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert result.returncode == 2
    assert not output.exists()
