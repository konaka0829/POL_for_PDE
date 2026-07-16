import json
import subprocess
import sys


def test_e0_cli_success_artifacts_and_overwrite_guard(tmp_path):
    out = tmp_path / "e0"
    command = [sys.executable, "scripts/paper1/run_e0.py", "--config", "configs/paper1_e0_smoke.json", "--output-dir", str(out), "--overwrite"]
    proc = subprocess.run(command, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    required = {"e0_summary.json", "reference_convergence.csv", "reference_convergence.json", "resampling_checks.json", "input_interface_checks.json", "model1_identity.json", "master_initial_conditions.pt", "master_manifest.json", "resolved_config.json", "environment.json"}
    assert required <= {p.name for p in out.iterdir()}
    summary = json.loads((out / "e0_summary.json").read_text())
    assert summary["status"] == "pass" and summary["num_failures"] == 0
    assert "NaN" not in (out / "e0_summary.json").read_text()
    env = json.loads((out / "environment.json").read_text())
    assert env["full_command"] and env["config_hash"] and env["master_archive_hash"]
    guard = subprocess.run(command[:-1], capture_output=True, text=True)
    assert guard.returncode != 0 and "--overwrite" in guard.stderr


def test_e0_cli_required_failure_returns_one(tmp_path):
    out = tmp_path / "failed_e0"
    proc = subprocess.run(
        [sys.executable, "scripts/paper1/run_e0.py", "--config", "configs/paper1_e0_failure.json", "--output-dir", str(out), "--overwrite"],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 1, proc.stdout + proc.stderr
    summary = json.loads((out / "e0_summary.json").read_text())
    assert summary["status"] == "fail" and summary["num_failures"] > 0
    assert summary["selected_reference"]["reference_nx"] is None
