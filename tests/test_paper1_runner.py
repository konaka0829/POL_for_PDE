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
from pol.paper1.runner import build_plan, execute_run, plan_to_dict


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

    def wait(self, timeout=None):
        return self.returncode

    def poll(self):
        return self.returncode

    def terminate(self):
        self.returncode = -15

    def kill(self):
        self.returncode = -9


def _temp_spec(tmp_path: Path, kind: str = "e0"):
    raw = json.loads(
        (ROOT / f"configs/runs/paper1_{kind}_smoke.json").read_text()
    )
    raw["run"] = {"name": "unit", "output_root": str(tmp_path / "runs")}
    source = tmp_path / "spec.json"
    source.write_text(json.dumps(raw))
    return load_run_spec(source, repo_root=ROOT)


def _write_owned_manifest(run_dir: Path, run_name: str) -> None:
    run_dir.mkdir(parents=True)
    (run_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "paper1-run-manifest-v1",
                "run_name": run_name,
                "run_dir": str(run_dir),
            }
        )
    )


def test_existing_guard_force_and_manifest(tmp_path: Path, monkeypatch) -> None:
    spec = _temp_spec(tmp_path)
    _write_owned_manifest(spec.run_dir, spec.name)
    marker = spec.run_dir / "marker"
    marker.write_text("old")
    sibling = spec.output_root / "sibling"
    sibling.mkdir()
    sibling_marker = sibling / "keep"
    sibling_marker.write_text("preserve")
    with pytest.raises(FileExistsError):
        execute_run(spec, repo_root=ROOT, force=False)
    monkeypatch.setattr(runner_module, "_git", lambda *args: "test")
    monkeypatch.setattr(runner_module.subprocess, "Popen", _FakeProcess)
    monkeypatch.setattr("pol.paper1.runner._verify_step", lambda step: None)
    assert execute_run(spec, repo_root=ROOT, force=True) == 0
    assert not marker.exists()
    assert sibling_marker.read_text() == "preserve"
    assert spec.output_root.exists()
    manifest = json.loads((spec.run_dir / "run_manifest.json").read_text())
    assert manifest["status"] == "pass"
    assert manifest["steps"][0]["returncode"] == 0


def test_child_failure_stops_and_records(tmp_path: Path, monkeypatch) -> None:
    spec = _temp_spec(tmp_path, "e2")
    calls = []

    class Failed(_FakeProcess):
        def __init__(self, command, **kwargs):
            super().__init__(command, **kwargs)
            self.returncode = 7
            calls.append(command)

    monkeypatch.setattr(runner_module, "_git", lambda *args: "test")
    monkeypatch.setattr(runner_module.subprocess, "Popen", Failed)
    assert execute_run(spec, repo_root=ROOT, force=False) == 1
    manifest = json.loads((spec.run_dir / "run_manifest.json").read_text())
    assert manifest["status"] == "fail"
    assert len(calls) == 1
    assert manifest["steps"][0]["returncode"] == 7
    assert manifest["steps"][0]["status"] == "fail"
    assert [step["status"] for step in manifest["steps"][1:]] == [
        "pending",
        "pending",
    ]


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
    with pytest.raises(ValueError, match="unsafe output root"):
        execute_run(unsafe, repo_root=ROOT, force=True)


def test_force_rejects_arbitrary_directory_without_deleting_marker(
    tmp_path: Path,
) -> None:
    spec = _temp_spec(tmp_path)
    spec.run_dir.mkdir(parents=True)
    marker = spec.run_dir / "important"
    marker.write_text("keep")
    with pytest.raises(ValueError, match="not runner-owned"):
        execute_run(spec, repo_root=ROOT, force=True)
    assert marker.read_text() == "keep"


def test_force_rejects_symlink_without_deleting_target(tmp_path: Path) -> None:
    spec = _temp_spec(tmp_path)
    spec.output_root.mkdir(parents=True)
    victim = spec.output_root / "victim"
    victim.mkdir()
    important = victim / "important.txt"
    important.write_text("keep")
    try:
        spec.run_dir.symlink_to(victim, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink creation unavailable: {exc}")
    with pytest.raises(ValueError, match="must not be a symlink"):
        execute_run(spec, repo_root=ROOT, force=True)
    assert important.read_text() == "keep"


def test_plan_rejects_unsafe_output_root_without_mutation(tmp_path: Path) -> None:
    spec = _temp_spec(tmp_path)
    unsafe = replace(spec, output_root=ROOT)
    before = set(ROOT.iterdir())
    with pytest.raises(ValueError, match="unsafe output root"):
        plan_to_dict(unsafe, repo_root=ROOT)
    assert set(ROOT.iterdir()) == before


def test_child_thread_environment_matches_step_policy(
    tmp_path: Path, monkeypatch
) -> None:
    spec = replace(_temp_spec(tmp_path, "e2"), torch_threads=3)
    environments = []

    class Capturing(_FakeProcess):
        def __init__(self, command, **kwargs):
            super().__init__(command, **kwargs)
            environments.append(kwargs["env"])

    monkeypatch.setattr(runner_module, "_git", lambda *args: "test")
    monkeypatch.setattr(runner_module.subprocess, "Popen", Capturing)
    monkeypatch.setattr(runner_module, "_verify_step", lambda step: None)
    assert execute_run(spec, repo_root=ROOT, force=False) == 0
    variables = (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    )
    assert [[env[name] for name in variables] for env in environments] == [
        ["1"] * 4,
        ["1"] * 4,
        ["3"] * 4,
    ]


def test_terminate_child_kills_after_timeout() -> None:
    class Stubborn:
        def __init__(self):
            self.returncode = None
            self.terminated = False
            self.killed = False
            self.wait_calls = 0

        def poll(self):
            return self.returncode

        def terminate(self):
            self.terminated = True

        def kill(self):
            self.killed = True
            self.returncode = -9

        def wait(self, timeout=None):
            self.wait_calls += 1
            if self.wait_calls == 1:
                raise subprocess.TimeoutExpired("child", timeout)
            return self.returncode

    process = Stubborn()
    returncode, failure = runner_module._terminate_child(process)
    assert process.terminated
    assert process.killed
    assert process.wait_calls == 2
    assert returncode == -9
    assert failure is None


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
