"""Exact artifact contracts and rollback-safe directory publication."""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import shutil
from typing import Callable, Iterable

from .io import file_sha256


def exact_artifact_tree(root: Path, expected: Iterable[str]) -> None:
    """Validate an exact regular-file tree and reject links/traversal."""
    wanted = set(expected)
    actual: set[str] = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"artifact tree contains symlink: {path}")
        if path.is_file():
            relative = path.relative_to(root).as_posix()
            if relative.startswith("../") or Path(relative).is_absolute():
                raise ValueError(f"unsafe artifact path: {relative}")
            actual.add(relative)
    if actual != wanted:
        raise ValueError(
            f"artifact tree mismatch: missing={sorted(wanted - actual)}, "
            f"extra={sorted(actual - wanted)}"
        )


def manifest_records(root: Path, names: Iterable[str]) -> list[dict[str, object]]:
    """Build sorted byte-level records for regular artifacts."""
    records = []
    for name in sorted(set(names)):
        path = root / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"missing or unsafe artifact: {name}")
        records.append(
            {
                "relative_path": name,
                "size_bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
    return records


@dataclass
class RunTransaction:
    """Directory-level staging, validation, publication, and rollback."""

    final_dir: Path

    def __post_init__(self) -> None:
        parent = self.final_dir.parent
        self.staging_dir = parent / f".{self.final_dir.name}.staging"
        self.backup_dir = parent / f".{self.final_dir.name}.backup"

    @staticmethod
    def _remove(path: Path) -> None:
        if path.is_symlink() or (path.exists() and not path.is_dir()):
            raise ValueError(f"unsafe transaction directory: {path}")
        if path.exists():
            shutil.rmtree(path)

    def begin(self) -> Path:
        self.final_dir.parent.mkdir(parents=True, exist_ok=True)
        for path in (self.staging_dir, self.backup_dir):
            self._remove(path)
        self.staging_dir.mkdir()
        return self.staging_dir

    def publish(self, validate: Callable[[Path], None]) -> None:
        validate(self.staging_dir)
        moved_old = False
        if self.final_dir.exists() or self.final_dir.is_symlink():
            if self.final_dir.is_symlink() or not self.final_dir.is_dir():
                raise ValueError(f"unsafe publication target: {self.final_dir}")
            os.replace(self.final_dir, self.backup_dir)
            moved_old = True
        try:
            os.replace(self.staging_dir, self.final_dir)
        except BaseException:
            if moved_old and self.backup_dir.exists() and not self.final_dir.exists():
                os.replace(self.backup_dir, self.final_dir)
            raise
        self._remove(self.backup_dir)

    def cleanup(self) -> None:
        self._remove(self.staging_dir)
