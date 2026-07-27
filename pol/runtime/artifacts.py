"""Exact artifact contracts and rollback-safe directory publication."""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import shutil
from typing import Callable, Iterable
import uuid

from .io import file_sha256


def exact_artifact_tree(root: Path, expected: Iterable[str]) -> None:
    """Validate an exact regular-file tree and reject links/traversal."""
    wanted = set(expected)
    actual: set[str] = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"artifact tree contains symlink: {path}")
        if path.is_dir():
            raise ValueError(f"artifact tree contains unexpected directory: {path}")
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
        token = uuid.uuid4().hex
        self.staging_dir = parent / f".{self.final_dir.name}.staging-{token}"
        self.backup_dir = parent / f".{self.final_dir.name}.backup-{token}"
        self._begun = False

    @staticmethod
    def _remove(path: Path) -> None:
        if path.is_symlink() or (path.exists() and not path.is_dir()):
            raise ValueError(f"unsafe transaction directory: {path}")
        if path.exists():
            shutil.rmtree(path)

    def begin(self) -> Path:
        if self._begun:
            raise RuntimeError("transaction already begun")
        self.final_dir.parent.mkdir(parents=True, exist_ok=True)
        self.staging_dir.mkdir()
        self._begun = True
        return self.staging_dir

    def publish(self, validate: Callable[[Path], None]) -> None:
        if not self._begun or not self.staging_dir.is_dir():
            raise RuntimeError("transaction has not begun")
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
        self._begun = False

    def cleanup(self) -> None:
        self._remove(self.staging_dir)
        # A backup can remain only if publication was interrupted after moving
        # the previous final. Restore it rather than deleting a known-good run.
        if self.backup_dir.exists() and not self.final_dir.exists():
            os.replace(self.backup_dir, self.final_dir)
        elif self.backup_dir.exists():
            self._remove(self.backup_dir)
        self._begun = False

    def archive_failure(self, validate: Callable[[Path], None]) -> Path:
        """Validate and retain diagnostics without replacing ``final_dir``."""
        if not self._begun or not self.staging_dir.is_dir():
            raise RuntimeError("transaction has not begun")
        validate(self.staging_dir)
        root = self.final_dir.parent / ".failed-attempts" / self.final_dir.name
        if root.is_symlink() or (root.exists() and not root.is_dir()):
            raise ValueError(f"unsafe failed-attempt root: {root}")
        root.mkdir(parents=True, exist_ok=True)
        destination = root / uuid.uuid4().hex
        os.replace(self.staging_dir, destination)
        self._begun = False
        return destination

    def __enter__(self) -> Path:
        return self.begin()

    def __exit__(self, exc_type, exc, traceback) -> None:
        if exc_type is not None:
            self.cleanup()


def execute_recipe_transaction(
    final_dir: Path,
    *,
    execute: Callable[[Path], object],
    validate_complete: Callable[[Path], object],
    validate_failure: Callable[[Path], object] | None = None,
) -> object:
    """Execute one recipe in sibling staging and publish only complete output."""
    from dataclasses import replace

    transaction = RunTransaction(final_dir)
    staging = transaction.begin()
    had_previous = False
    if final_dir.is_dir():
        try:
            validate_complete(final_dir)
            had_previous = True
        except Exception:
            # Invalid/stale output has no preservation privilege. It is still
            # replaced transactionally, never deleted before staging is ready.
            had_previous = False
    try:
        result = execute(staging)
        status = getattr(result, "status", None)
        published_output: Path | None
        if status == "pass":
            transaction.publish(validate_complete)
            published_output = final_dir
        elif had_previous:
            published_output = (
                transaction.archive_failure(validate_failure)
                if validate_failure is not None
                else None
            )
            if published_output is None:
                transaction.cleanup()
        else:
            # A first failed attempt is publishable only when the recipe owns
            # and validates an exact diagnostic contract. Never expose an
            # arbitrary partial staging tree as the requested output.
            if validate_failure is None:
                transaction.cleanup()
                published_output = None
            else:
                transaction.publish(validate_failure)
                published_output = final_dir
        summary_path = getattr(result, "summary_path", None)
        return replace(
            result,
            output_dir=published_output or final_dir,
            summary_path=(
                published_output / Path(summary_path).name
                if summary_path is not None and published_output is not None
                else None
            ),
        )
    except BaseException:
        transaction.cleanup()
        raise
