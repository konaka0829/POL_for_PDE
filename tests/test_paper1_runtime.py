from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from pol.runtime.recipe import numerical_thread_scope


ROOT = Path(__file__).resolve().parents[1]
THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def test_shared_atomic_io_and_transaction_preserve_previous_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from pol.runtime.artifacts import RunTransaction, exact_artifact_tree
    from pol.runtime.io import write_csv, write_strict_json

    final = tmp_path / "run"
    final.mkdir()
    (final / "old.txt").write_text("old", encoding="utf-8")
    transaction = RunTransaction(final)
    staging = transaction.begin()
    write_strict_json(staging / "record.json", {"finite": 1.0})
    write_csv(staging / "table.csv", [{"id": 1, "value": 2.0}])

    import pol.runtime.artifacts as artifacts

    real_replace = artifacts.os.replace

    def fail_staging(source: Path, destination: Path) -> None:
        if Path(source) == transaction.staging_dir:
            raise OSError("fault injection")
        real_replace(source, destination)

    monkeypatch.setattr(artifacts.os, "replace", fail_staging)
    with pytest.raises(OSError, match="fault injection"):
        transaction.publish(
            lambda root: exact_artifact_tree(root, {"record.json", "table.csv"})
        )
    assert (final / "old.txt").read_text(encoding="utf-8") == "old"


def test_failed_rerun_archives_exact_diagnostics_and_preserves_pass(
    tmp_path: Path,
) -> None:
    from pol.runtime.artifacts import execute_recipe_transaction, exact_artifact_tree
    from pol.runtime.recipe import RecipeResult

    final = tmp_path / "run"
    final.mkdir()
    (final / "pass.json").write_text('{"status":"pass"}\n')

    def execute(staging: Path) -> RecipeResult:
        summary = staging / "failure.json"
        summary.write_text('{"status":"fail"}\n')
        return RecipeResult("fail", 1, staging, {"status": "fail"}, summary)

    result = execute_recipe_transaction(
        final,
        execute=execute,
        validate_complete=lambda root: exact_artifact_tree(root, {"pass.json"}),
        validate_failure=lambda root: exact_artifact_tree(root, {"failure.json"}),
    )
    assert (final / "pass.json").read_text() == '{"status":"pass"}\n'
    assert result.output_dir != final
    assert result.output_dir.is_relative_to(tmp_path / ".failed-attempts")
    assert result.summary_path == result.output_dir / "failure.json"


@pytest.mark.parametrize("failure", [None, RuntimeError, KeyboardInterrupt])
def test_numerical_thread_scope_sets_and_restores(monkeypatch, failure) -> None:
    previous_torch = torch.get_num_threads()
    for index, name in enumerate(THREAD_VARIABLES):
        if index % 2:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, str(index + 7))
    previous_environment = {name: os.environ.get(name) for name in THREAD_VARIABLES}

    def invoke() -> None:
        with numerical_thread_scope(2):
            assert torch.get_num_threads() == 2
            assert {os.environ[name] for name in THREAD_VARIABLES} == {"2"}
            if failure is not None:
                raise failure("scope failure")

    if failure is None:
        invoke()
    else:
        with pytest.raises(failure):
            invoke()
    assert torch.get_num_threads() == previous_torch
    assert {name: os.environ.get(name) for name in THREAD_VARIABLES} == (
        previous_environment
    )


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_numerical_thread_scope_rejects_invalid_count(value) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        with numerical_thread_scope(value):
            pass


def test_numerical_thread_scope_restores_environment_if_initial_set_fails(
    monkeypatch,
) -> None:
    previous_environment = {name: os.environ.get(name) for name in THREAD_VARIABLES}

    def fail(_thread_count: int) -> None:
        raise RuntimeError("synthetic thread initialization failure")

    monkeypatch.setattr(torch, "set_num_threads", fail)
    with pytest.raises(RuntimeError, match="synthetic"):
        with numerical_thread_scope(2):
            pass
    assert {name: os.environ.get(name) for name in THREAD_VARIABLES} == (
        previous_environment
    )


def test_importing_paper1_config_does_not_load_legacy_modules() -> None:
    code = """
import sys
import pol.paper1.config
for name in ("pol.model123_1d", "pol.reservoir_1d", "pol.elm"):
    assert name not in sys.modules, name
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_import_pol_has_no_legacy_public_api() -> None:
    import pol

    for name in (
        "FixedRandomELM",
        "DatasetConfig",
        "Reservoir1DSolver",
        "simulate_burgers_split_step",
    ):
        assert not hasattr(pol, name)
