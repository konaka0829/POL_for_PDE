from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import zipfile


ROOT = Path(__file__).resolve().parents[1]


def test_wheel_contains_all_pol_packages(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            ".",
            "--no-deps",
            "--no-build-isolation",
            "--wheel-dir",
            str(tmp_path),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    wheels = list(tmp_path.glob("*.whl"))
    assert len(wheels) == 1
    with zipfile.ZipFile(wheels[0]) as archive:
        names = set(archive.namelist())
    for package in (
        "pol",
        "pol/model123_1d",
        "pol/paper1",
        "pol/paper1/recipes",
        "pol/paper1/matrix_plugins",
        "pol/paper1/plot_recipes",
        "pol/plots",
        "pol/runtime",
        "pol/workflow",
    ):
        assert f"{package}/__init__.py" in names
