from __future__ import annotations

import os
import subprocess
from pathlib import Path


def subprocess_env_one_thread() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("NUMEXPR_NUM_THREADS", "1")
    env.setdefault("TORCH_NUM_THREADS", "1")
    return env


def run_cli(cmd: list[str], cwd: Path, timeout: int = 60) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=cwd,
        capture_output=True,
        text=True,
        env=subprocess_env_one_thread(),
        timeout=timeout,
        check=False,
    )
