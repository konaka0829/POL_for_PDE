from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from pol.paper1.regression_baseline import (
    build_e1_matrix_scientific_baseline,
)
from pol.paper1.scientific_comparison import (
    assert_scientific_record_matches,
    policy_from_baseline,
)


ROOT = Path(__file__).resolve().parents[1]
EXPECTED = json.loads(
    (
        ROOT / "tests/fixtures/paper1_e1_matrix_smoke_baseline_v2.json"
    ).read_text(encoding="utf-8")
)


def _environment() -> dict[str, str]:
    environment = os.environ.copy()
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        environment[name] = "1"
    return environment


def _run(arguments: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *arguments],
        cwd=ROOT,
        env=_environment(),
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    )


def _spec(tmp_path: Path, name: str, jobs: int) -> Path:
    raw = json.loads(
        (
            ROOT / "configs/runs/paper1_e1_resolution_sweep_smoke.json"
        ).read_text()
    )
    raw["run"] = {"name": name, "output_root": str(tmp_path)}
    raw["execution"]["jobs"] = jobs
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    return path


def _assert_baseline(run_dir: Path) -> None:
    manifest = json.loads((run_dir / "matrix_manifest.json").read_text())
    assert manifest["status"] == "pass"
    actual = build_e1_matrix_scientific_baseline(
        run_dir, source_revision="runtime"
    )
    assert_scientific_record_matches(
        actual["record"],
        EXPECTED["record"],
        policy_from_baseline(EXPECTED),
    )


@pytest.mark.slow
def test_matrix_jobs_are_deterministic_and_match_legacy_baseline(
    tmp_path: Path,
) -> None:
    first_spec = _spec(tmp_path, "jobs1", 1)
    second_spec = _spec(tmp_path, "jobs2", 2)
    for spec in (first_spec, second_spec):
        result = _run(["-m", "pol", "run", str(spec)])
        assert result.returncode == 0, result.stdout + result.stderr
        result = _run(["-m", "pol", "run", str(spec), "--plots-only"])
        assert result.returncode == 0, result.stdout + result.stderr
    first = tmp_path / "jobs1"
    second = tmp_path / "jobs2"
    _assert_baseline(first)
    _assert_baseline(second)
    for run_dir in (first, second):
        manifest = json.loads((run_dir / "matrix_manifest.json").read_text())
        assert manifest["compute_status"] == "pass"
        assert manifest["plot_status"] == "pass"
        assert manifest["plot_tasks"][0]["executed_or_reused"] == "reused"
    for name in (
        "sweep_selected_results.csv",
        "sweep_readout_diagnostics.csv",
        "sweep_noise_summary.csv",
    ):
        assert (first / "aggregate" / name).read_bytes() == (
            second / "aggregate" / name
        ).read_bytes()
    first_rows = json.loads((first / "matrix_plan.json").read_text())["cells"]
    second_rows = json.loads((second / "matrix_plan.json").read_text())["cells"]
    assert [
        (row["run_index"], row["cell_id"], row["experiment_memberships"])
        for row in first_rows
    ] == [
        (row["run_index"], row["cell_id"], row["experiment_memberships"])
        for row in second_rows
    ]


@pytest.mark.slow
def test_matrix_resume_reruns_only_missing_or_tampered_cells(
    tmp_path: Path,
) -> None:
    spec = _spec(tmp_path, "resume", 2)
    assert _run(["-m", "pol", "run", str(spec)]).returncode == 0
    run_dir = tmp_path / "resume"
    assert _run(["-m", "pol", "run", str(spec)]).returncode == 0
    manifest = json.loads((run_dir / "matrix_manifest.json").read_text())
    assert manifest["dependencies"]["e0"]["executed_or_reused"] == "reused"
    assert {cell["executed_or_reused"] for cell in manifest["cells"]} == {"reused"}

    first = Path(manifest["cells"][0]["output_dir"])
    with (first / "selected_results.csv").open("a", encoding="utf-8") as handle:
        handle.write("tamper\n")
    assert _run(["-m", "pol", "run", str(spec)]).returncode == 0
    manifest = json.loads((run_dir / "matrix_manifest.json").read_text())
    assert [cell["executed_or_reused"] for cell in manifest["cells"]].count(
        "executed"
    ) == 1

    second = Path(manifest["cells"][1]["output_dir"])
    (second / "noise_summary.csv").unlink()
    assert _run(["-m", "pol", "run", str(spec)]).returncode == 0
    manifest = json.loads((run_dir / "matrix_manifest.json").read_text())
    assert [cell["executed_or_reused"] for cell in manifest["cells"]].count(
        "executed"
    ) == 1
    _assert_baseline(run_dir)
