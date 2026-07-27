from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import zipfile


ROOT = Path(__file__).resolve().parents[1]


def test_wheel_contains_all_pol_packages(tmp_path: Path) -> None:
    source = tmp_path / "source"
    shutil.copytree(
        ROOT,
        source,
        ignore=shutil.ignore_patterns(
            ".git",
            ".codex",
            "build",
            "dist",
            "outputs",
            "outputs_paper1",
            "tex",
            "*.egg-info",
            "__pycache__",
            "*.pyc",
        ),
    )
    cwd = tmp_path / "build-cwd"
    cwd.mkdir()
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "build",
            "--wheel",
            "--no-isolation",
            "--outdir",
            str(tmp_path),
            str(source.resolve()),
        ],
        cwd=cwd,
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
        "pol/numerics",
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
    assert not any(
        fragment in name
        for name in names
        for fragment in (
            "model123",
            "burgers_spectral_1d.py",
            "spectral_etdrk4_1d.py",
        )
    )


def test_sdist_contains_active_package_and_excludes_repository_material(
    tmp_path: Path,
) -> None:
    cwd = tmp_path / "build-cwd"
    cwd.mkdir()
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "build",
            "--sdist",
            "--no-isolation",
            "--outdir",
            str(tmp_path),
            str(ROOT),
        ],
        cwd=cwd,
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
        "README.md",
        "pol/__init__.py",
        "pol/numerics/__init__.py",
        "pol/paper1/__init__.py",
        "pol/plots/__init__.py",
        "pol/runtime/__init__.py",
        "pol/workflow/__init__.py",
    ):
        assert f"{root}/{required}" in names
    assert not any(
        "__pycache__" in name
        or name.endswith((".pyc", ".pyo"))
        or "/.pytest_cache/" in name
        or "/outputs_paper1/" in name
        or "/build/" in name
        or f"{root}/configs/" in name
        or f"{root}/docs/" in name
        or f"{root}/scripts/" in name
        or f"{root}/tests/" in name
        for name in names
    )
    assert not any(
        fragment in name
        for name in names
        for fragment in (
            "model123",
            "configs/B0_smoke.json",
            "configs/B1_burgers_grf.json",
            "burgers_spectral_1d.py",
            "spectral_etdrk4_1d.py",
        )
    )
