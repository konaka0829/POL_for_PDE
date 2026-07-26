"""Runtime types and scoped numerical thread control for recipes."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Iterator, Literal, Mapping


_THREAD_ENVIRONMENT_VARIABLES = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


@dataclass(frozen=True)
class RecipeInvocation:
    """Provenance and runtime context supplied to an experiment recipe."""

    repo_root: Path
    working_directory: Path
    command: tuple[str, ...]
    torch_threads: int = 1


@dataclass(frozen=True)
class RecipeResult:
    """Outcome returned by an import-safe experiment recipe."""

    status: Literal["pass", "fail"]
    exit_code: int
    output_dir: Path
    console_payload: Mapping[str, Any]
    summary_path: Path | None = None
    reused_complete_output: bool = False


class RecipeUsageError(ValueError):
    """CLI/preflight input error corresponding to argparse exit code 2."""


def validate_recursive_delete_target(
    output_path: Path,
    *,
    repo_root: Path,
    protected_paths: tuple[Path, ...] = (),
) -> None:
    """Reject symlinked, special, or input-containing recursive delete targets."""
    parent = output_path.parent.resolve()
    candidate = parent / output_path.name
    filesystem_root = Path(candidate.anchor).resolve()
    protected = {
        filesystem_root,
        Path.home().resolve(),
        repo_root.resolve(),
        repo_root.resolve().parent,
        *(path.resolve() for path in protected_paths),
    }
    if candidate.is_symlink():
        raise ValueError(f"output path must not be a symlink: {output_path}")
    if candidate.resolve(strict=False) != candidate:
        raise ValueError(f"output path resolves unexpectedly: {output_path}")
    for path in protected:
        if path == candidate or path.is_relative_to(candidate):
            raise ValueError(
                f"output path contains protected path {path}: {output_path}"
            )


@contextmanager
def numerical_thread_scope(thread_count: int) -> Iterator[None]:
    """Temporarily set numerical environment and PyTorch intra-op threads."""
    if isinstance(thread_count, bool) or not isinstance(thread_count, int):
        raise ValueError("thread_count must be a positive integer")
    if thread_count <= 0:
        raise ValueError("thread_count must be a positive integer")
    previous_environment = {
        name: os.environ.get(name) for name in _THREAD_ENVIRONMENT_VARIABLES
    }
    torch_module: Any | None = None
    previous_torch_threads: int | None = None
    torch_threads_changed = False
    try:
        for name in _THREAD_ENVIRONMENT_VARIABLES:
            os.environ[name] = str(thread_count)

        import torch

        torch_module = torch
        previous_torch_threads = torch.get_num_threads()
        torch.set_num_threads(thread_count)
        torch_threads_changed = True
        yield
    finally:
        try:
            if (
                torch_threads_changed
                and torch_module is not None
                and previous_torch_threads is not None
            ):
                torch_module.set_num_threads(previous_torch_threads)
        finally:
            for name, previous in previous_environment.items():
                if previous is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = previous
