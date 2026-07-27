from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import tarfile
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
    assert not any(
        "__pycache__" in name
        or name.endswith((".pyc", ".pyo"))
        or "/.pytest_cache/" in name
        or name.startswith(("outputs_paper1/", "build/"))
        for name in names
    )
    assert not any("/scripts/paper1/" in name for name in names)


def test_sdist_contains_research_specs_and_is_clean(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "build",
            "--sdist",
            "--no-isolation",
            "--outdir",
            str(tmp_path),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    archives = list(tmp_path.glob("*.tar.gz"))
    assert len(archives) == 1
    with tarfile.open(archives[0], "r:gz") as archive:
        names = {member.name for member in archive.getmembers()}
    root = next(name.split("/", 1)[0] for name in names)
    for required in (
        "AGENTS.md",
        "README.md",
        "configs/runs/paper1_e0_smoke.json",
        "configs/runs/paper1_e1_smoke.json",
        "configs/runs/paper1_e2_smoke.json",
        "configs/runs/paper1_e1_resolution_sweep_smoke.json",
        "docs/legacy_removed.md",
    ):
        assert f"{root}/{required}" in names
    assert not any(
        "__pycache__" in name
        or name.endswith((".pyc", ".pyo"))
        or "/.pytest_cache/" in name
        or "/outputs_paper1/" in name
        or "/build/" in name
        for name in names
    )
    assert not any("/scripts/paper1/" in name for name in names)
