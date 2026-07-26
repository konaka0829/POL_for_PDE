from __future__ import annotations

import io
import json
from dataclasses import replace
from pathlib import Path
import subprocess
import sys

import pytest

from pol.cli import main
import pol.paper1.runner as runner_module
from pol.paper1.run_spec import load_run_spec
from pol.paper1.runner import build_plan, execute_run


ROOT = Path(__file__).resolve().parents[1]


def _spec(kind: str):
    return load_run_spec(ROOT / f"configs/runs/paper1_{kind}_smoke.json", repo_root=ROOT)


@pytest.mark.parametrize(
    ("kind", "names"), [("e0", ["e0"]), ("e1", ["e0", "e1"]), ("e2", ["e0", "master_dataset", "e2"])]
)
def test_plans_use_existing_scripts(kind: str, names: list[str]) -> None:
    plan = build_plan(_spec(kind), repo_root=ROOT)
    assert [step.name for step in plan] == names
    assert all(step.command[0] == sys.executable for step in plan)
    assert all("scripts/paper1/" in step.command[1] for step in plan)


@pytest.mark.parametrize("kind", ["e1", "e2"])
def test_skip_plots_is_conditional(kind: str) -> None:
    spec = _spec(kind)
    assert "--skip-plots" not in build_plan(spec, repo_root=ROOT)[-1].command
    enabled = spec.__class__(**{**spec.__dict__, "skip_plots": True})
    assert "--skip-plots" in build_plan(enabled, repo_root=ROOT)[-1].command


def test_plan_has_no_filesystem_or_popen(tmp_path: Path, monkeypatch, capsys) -> None:
    raw = json.loads((ROOT / "configs/runs/paper1_e0_smoke.json").read_text())
    raw["run"]["output_root"] = str(tmp_path / "out")
    source = tmp_path / "spec.json"
    source.write_text(json.dumps(raw))
    monkeypatch.setattr(runner_module.subprocess, "Popen", lambda *a, **k: pytest.fail("Popen called"))
    assert main(["run", str(source), "--plan"]) == 0
    assert not (tmp_path / "out").exists()
    assert json.loads(capsys.readouterr().out)["schema_version"] == "paper1-run-plan-v1"


class _FakeProcess:
    def __init__(self, command, **kwargs):
        self.command = command
        self.stdout = io.StringIO("child output\n")
        self.returncode = 0

    def wait(self):
        return self.returncode

    def poll(self):
        return self.returncode

    def terminate(self):
        self.returncode = -15


def _temp_spec(tmp_path: Path):
    raw = json.loads((ROOT / "configs/runs/paper1_e0_smoke.json").read_text())
    raw["run"] = {"name": "unit", "output_root": str(tmp_path / "runs")}
    source = tmp_path / "spec.json"
    source.write_text(json.dumps(raw))
    return load_run_spec(source, repo_root=ROOT)


def test_existing_guard_force_and_manifest(tmp_path: Path, monkeypatch) -> None:
    spec = _temp_spec(tmp_path)
    spec.run_dir.mkdir(parents=True)
    marker = spec.run_dir / "marker"
    marker.write_text("old")
    with pytest.raises(FileExistsError):
        execute_run(spec, repo_root=ROOT, force=False)
    monkeypatch.setattr(runner_module, "_git", lambda *args: "test")
    monkeypatch.setattr(runner_module.subprocess, "Popen", _FakeProcess)
    monkeypatch.setattr("pol.paper1.runner._verify_step", lambda step: None)
    assert execute_run(spec, repo_root=ROOT, force=True) == 0
    assert not marker.exists()
    assert spec.output_root.exists()
    manifest = json.loads((spec.run_dir / "run_manifest.json").read_text())
    assert manifest["status"] == "pass"
    assert manifest["steps"][0]["returncode"] == 0


def test_child_failure_stops_and_records(tmp_path: Path, monkeypatch) -> None:
    spec = _temp_spec(tmp_path)

    class Failed(_FakeProcess):
        def __init__(self, command, **kwargs):
            super().__init__(command, **kwargs)
            self.returncode = 7

    monkeypatch.setattr(runner_module, "_git", lambda *args: "test")
    monkeypatch.setattr(runner_module.subprocess, "Popen", Failed)
    assert execute_run(spec, repo_root=ROOT, force=False) == 1
    manifest = json.loads((spec.run_dir / "run_manifest.json").read_text())
    assert manifest["status"] == "fail"
    assert manifest["steps"][0]["returncode"] == 7
    assert manifest["steps"][0]["status"] == "fail"


def test_exit_zero_missing_summary_fails(tmp_path: Path, monkeypatch) -> None:
    spec = _temp_spec(tmp_path)
    monkeypatch.setattr(runner_module, "_git", lambda *args: "test")
    monkeypatch.setattr(runner_module.subprocess, "Popen", _FakeProcess)
    assert execute_run(spec, repo_root=ROOT, force=False) == 1
    manifest = json.loads((spec.run_dir / "run_manifest.json").read_text())
    assert "missing saved output" in manifest["failure"]


def test_unsafe_run_directory_is_rejected(tmp_path: Path) -> None:
    spec = _temp_spec(tmp_path)
    unsafe = replace(spec, output_root=ROOT.parent, name=ROOT.name)
    with pytest.raises(ValueError, match="unsafe run directory"):
        execute_run(unsafe, repo_root=ROOT, force=True)


def test_keyboard_interrupt_terminates_and_returns_130(tmp_path: Path, monkeypatch) -> None:
    spec = _temp_spec(tmp_path)
    processes = []

    class Interrupting(_FakeProcess):
        def __init__(self, command, **kwargs):
            super().__init__(command, **kwargs)
            self.stdout = self
            self.terminated = False
            self.returncode = None
            processes.append(self)

        def __iter__(self):
            raise KeyboardInterrupt

        def terminate(self):
            self.terminated = True
            self.returncode = -15

    monkeypatch.setattr(runner_module, "_git", lambda *args: "test")
    monkeypatch.setattr(runner_module.subprocess, "Popen", Interrupting)
    assert execute_run(spec, repo_root=ROOT, force=False) == 130
    assert processes[0].terminated
    manifest = json.loads((spec.run_dir / "run_manifest.json").read_text())
    assert manifest["status"] == "interrupted"
    assert manifest["steps"][0]["status"] == "interrupted"
    assert manifest["steps"][0]["returncode"] == -15
    assert manifest["steps"][0]["duration_seconds"] is not None


def test_stream_failure_terminates_child_and_records_returncode(
    tmp_path: Path, monkeypatch
) -> None:
    spec = _temp_spec(tmp_path)
    processes = []

    class BrokenStream:
        def __iter__(self):
            raise OSError("simulated stream read failure")

    class Running(_FakeProcess):
        def __init__(self, command, **kwargs):
            super().__init__(command, **kwargs)
            self.stdout = BrokenStream()
            self.terminated = False
            self.returncode = None
            processes.append(self)

        def terminate(self):
            self.terminated = True
            self.returncode = -15

    monkeypatch.setattr(runner_module, "_git", lambda *args: "test")
    monkeypatch.setattr(runner_module.subprocess, "Popen", Running)
    assert execute_run(spec, repo_root=ROOT, force=False) == 1
    assert processes[0].terminated
    manifest = json.loads((spec.run_dir / "run_manifest.json").read_text())
    assert manifest["status"] == "fail"
    assert manifest["steps"][0]["status"] == "fail"
    assert manifest["steps"][0]["returncode"] == -15
    assert manifest["steps"][0]["duration_seconds"] is not None
    assert "simulated stream read failure" in manifest["failure"]


@pytest.mark.parametrize(
    "arguments", [["--help"], ["run", "--help"]]
)
def test_cli_help(arguments: list[str]) -> None:
    result = subprocess.run(
        [sys.executable, "-m", "pol", *arguments],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "usage:" in result.stdout
