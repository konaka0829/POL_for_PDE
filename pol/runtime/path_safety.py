"""Shared lexical path validation for runner-owned output directories."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable


def resolve_safe_run_directory(
    *,
    name: str,
    output_root: Path,
    repo_root: Path,
    protected_paths: Iterable[Path],
) -> tuple[Path, Path]:
    """Return a safe resolved root and non-symlink lexical child run path."""
    root = repo_root.resolve()
    resolved_output_root = output_root.resolve()
    filesystem_root = Path(resolved_output_root.anchor).resolve()
    forbidden_roots = {
        filesystem_root,
        Path.home().resolve(),
        root,
        root.parent,
    }
    if resolved_output_root in forbidden_roots:
        raise ValueError(f"unsafe output root: {resolved_output_root}")
    run_dir = resolved_output_root / name
    forbidden_run_dirs = {*forbidden_roots, resolved_output_root}
    if run_dir.parent != resolved_output_root or run_dir in forbidden_run_dirs:
        raise ValueError(f"unsafe run directory: {run_dir}")
    if run_dir.is_symlink():
        raise ValueError(f"run directory must not be a symlink: {run_dir}")
    resolved_target = run_dir.resolve(strict=False)
    if (
        resolved_target != run_dir
        or resolved_target.parent != resolved_output_root
    ):
        raise ValueError(f"run directory escapes output root: {run_dir}")
    protected = {
        root,
        Path.home().resolve(),
        *(path.resolve() for path in protected_paths),
    }
    for path in protected:
        if path == run_dir or path.is_relative_to(run_dir):
            raise ValueError(
                f"run directory contains protected path {path}: {run_dir}"
            )
    return resolved_output_root, run_dir
