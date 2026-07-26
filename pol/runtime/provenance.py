"""Small provenance helpers with explicit repository roots."""
from __future__ import annotations

from pathlib import Path
import subprocess
from typing import Sequence


def git_output(repo_root: Path, arguments: Sequence[str]) -> str:
    """Run a read-only Git command and return stdout or ``unknown``."""
    try:
        process = subprocess.run(
            ["git", *arguments],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return "unknown"
    return process.stdout.strip() if process.returncode == 0 else "unknown"
