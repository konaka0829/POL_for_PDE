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


def test_all_legacy_exports_remain_lazy_compatible() -> None:
    import pol

    for name in pol.__all__:
        assert getattr(pol, name) is not None
