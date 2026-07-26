from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from pol.paper1.config import canonical_config_json, load_config_json
from pol.paper1.regression_baseline import (
    canonicalize_scientific,
    semantic_model_digest,
)


ROOT = Path(__file__).resolve().parents[1]
EXPECTED = json.loads(
    (
        ROOT / "tests/fixtures/paper1_e1_sweep_smoke_baseline_v1.json"
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


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_hash(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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
    generated = sorted(
        hashlib.sha256(canonical_config_json(load_config_json(path)).encode()).hexdigest()
        for path in (run_dir / "generated_configs").glob("*.json")
    )
    assert generated == EXPECTED["generated_config_canonical_sha256"]
    for cell in manifest["cells"]:
        expected = EXPECTED["cells"][cell["human_slug"]]
        output = Path(cell["output_dir"])
        summary = canonicalize_scientific(
            json.loads((output / "e1_summary.json").read_text())
        )
        assert _canonical_hash(summary) == expected["summary_semantic_sha256"]
        assert semantic_model_digest(output / "selected_models.pt") == expected[
            "selected_models_semantic_sha256"
        ]
        assert {
            path.name: _sha(path) for path in sorted(output.glob("*.csv"))
        } == expected["csv_sha256"]
    assert {
        path.name: _sha(path)
        for path in sorted((run_dir / "aggregate").glob("sweep_*.csv"))
    } == EXPECTED["aggregate_csv_sha256"]


@pytest.mark.slow
def test_matrix_jobs_are_deterministic_and_match_legacy_baseline(
    tmp_path: Path,
) -> None:
    first_spec = _spec(tmp_path, "jobs1", 1)
    second_spec = _spec(tmp_path, "jobs2", 2)
    for spec in (first_spec, second_spec):
        result = _run(["-m", "pol", "run", str(spec)])
        assert result.returncode == 0, result.stdout + result.stderr
    first = tmp_path / "jobs1"
    second = tmp_path / "jobs2"
    _assert_baseline(first)
    _assert_baseline(second)
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
    assert manifest["e0"]["executed_or_reused"] == "reused"
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


@pytest.mark.slow
def test_legacy_wrapper_uses_matrix_engine_and_preserves_plots(
    tmp_path: Path,
) -> None:
    e0 = tmp_path / "e0"
    result = _run(
        [
            "scripts/paper1/run_e0.py",
            "--config",
            "configs/paper1_e0_for_e1_smoke.json",
            "--output-dir",
            str(e0),
            "--overwrite",
        ]
    )
    assert result.returncode == 0, result.stdout + result.stderr
    output = tmp_path / "legacy"
    result = _run(
        [
            "scripts/paper1/run_e1_sweep.py",
            "--base-config",
            "configs/paper1_e1_smoke.json",
            "--sweep-spec",
            "configs/paper1_e1_sweep_smoke.json",
            "--e0-dir",
            str(e0),
            "--output-root",
            str(output),
            "--jobs",
            "2",
            "--torch-threads",
            "1",
        ]
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stderr.count("deprecated") == 1
    manifest = json.loads((output / "sweep_plot_manifest.json").read_text())
    plots = sorted(
        [record.get("relative_path"), record.get("format"), record["status"]]
        for record in manifest["plots"]
    )
    assert plots == EXPECTED["plots"]
    for name in (
        "sweep_selected_results.csv",
        "sweep_readout_diagnostics.csv",
        "sweep_noise_summary.csv",
    ):
        assert _sha(output / name) == EXPECTED["aggregate_csv_sha256"][name]
