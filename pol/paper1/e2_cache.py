"""Versioned, atomic tensor caches used by Paper 1 E2."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import stat
import tempfile
from typing import Any, Callable

import torch

CACHE_SCHEMA_VERSION = "paper1-e2-cache-v3"


def canonical_object(value: Any) -> Any:
    """Return the JSON-stable representation used for keys and metadata."""
    if isinstance(value, dict):
        return {str(k): canonical_object(value[k]) for k in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [canonical_object(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("cache keys must contain only finite numbers")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"unsupported cache-key value: {type(value).__name__}")


def stable_hash(value: Any) -> str:
    data = json.dumps(canonical_object(value), sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode()
    return hashlib.sha256(data).hexdigest()


def tensor_hash(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    h = hashlib.sha256()
    h.update(str(tensor.dtype).encode())
    h.update(json.dumps(list(tensor.shape)).encode())
    h.update(tensor.numpy().tobytes())
    return h.hexdigest()


def _atomic_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_json(path: Path, value: Any) -> None:
    _atomic_bytes(path, (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode())


class TensorCache:
    """Content-addressed cache with canonical keys and complete read-back QA."""

    def __init__(self, root: Path, *, resume: bool):
        self.root, self.resume = Path(root), resume
        self.stats = {
            "states": {"hits": 0, "misses": 0, "solver_invocations": 0},
            "features": {"hits": 0, "misses": 0},
        }

    @staticmethod
    def _type(path: Path) -> str:
        try:
            mode = os.lstat(path).st_mode
        except FileNotFoundError:
            return "missing"
        if stat.S_ISLNK(mode):
            return "symlink"
        if stat.S_ISREG(mode):
            return "file"
        if stat.S_ISDIR(mode):
            return "directory"
        return "special"

    def _safe_directory(self, path: Path, *, create: bool) -> None:
        kind = self._type(path)
        if kind == "missing" and create:
            path.mkdir(parents=True, exist_ok=False)
            kind = self._type(path)
        if kind != "directory":
            raise ValueError(f"unsafe cache directory ({kind}): {path}")
        if path.resolve(strict=True) != path.absolute():
            raise ValueError(f"cache directory traverses a symlink: {path}")

    def _safe_unit_type(self, path: Path, *, allow_missing: bool = True) -> str:
        kind = self._type(path)
        if kind == "missing" and allow_missing:
            return kind
        if kind != "file":
            raise ValueError(f"unsafe cache unit path ({kind}): {path}")
        if path.resolve(strict=True).parent != path.parent.resolve(strict=True):
            raise ValueError(f"cache unit resolves outside expected directory: {path}")
        return kind

    def _remove_unsafe_unit(self, path: Path) -> None:
        kind = self._type(path)
        if kind == "missing":
            return
        if kind in {"file", "symlink"}:
            path.unlink()
            return
        if kind == "directory":
            try:
                path.rmdir()
            except OSError as exc:
                raise ValueError(
                    f"refusing to remove non-empty unsafe cache directory: {path}"
                ) from exc
            return
        raise ValueError(f"refusing to remove special cache path: {path}")

    def _inspect_unit(self, paths: tuple[Path, ...]) -> tuple[bool, ...]:
        result = []
        for path in paths:
            kind = self._type(path)
            if kind not in {"missing", "file"}:
                raise ValueError(f"unsafe cache unit path ({kind}): {path}")
            result.append(kind == "file")
        return tuple(result)

    @property
    def hits(self) -> int:
        return sum(int(v["hits"]) for v in self.stats.values())

    @property
    def misses(self) -> int:
        return sum(int(v["misses"]) for v in self.stats.values())

    def _load(self, path: Path, meta_path: Path, key: Any, digest: str, kind: str, *, count_hit: bool = True):
        meta = json.loads(meta_path.read_text())
        if meta.get("schema_version") != CACHE_SCHEMA_VERSION:
            raise ValueError("cache schema mismatch")
        if meta.get("key") != key or meta.get("key_sha256") != stable_hash(key):
            raise ValueError("cache key mismatch")
        if hashlib.sha256(path.read_bytes()).hexdigest() != meta.get("file_sha256"):
            raise ValueError("cache file hash mismatch")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        values = payload.get("values")
        if not isinstance(values, torch.Tensor) or not bool(torch.isfinite(values).all()):
            raise ValueError("cache tensor missing or non-finite")
        if tensor_hash(values) != meta.get("tensor_hash"):
            raise ValueError("cache tensor hash mismatch")
        if list(values.shape) != meta.get("shape") or str(values.dtype) != meta.get("dtype"):
            raise ValueError("cache tensor shape/dtype mismatch")
        if canonical_object(payload.get("solver_metadata", {})) != meta.get("solver_metadata"):
            raise ValueError("cache solver metadata mismatch")
        if count_hit:
            self.stats[kind]["hits"] += 1
        return values, payload.get("solver_metadata", {}), digest

    def get_or_compute(
        self, kind: str, key: dict[str, Any],
        compute: Callable[[], tuple[torch.Tensor, dict[str, Any]]],
    ) -> tuple[torch.Tensor, dict[str, Any], str]:
        if kind not in self.stats:
            raise ValueError(f"unknown cache kind: {kind}")
        canonical = canonical_object(key)
        digest = stable_hash({"schema": CACHE_SCHEMA_VERSION, "kind": kind, "key": canonical})
        directory = self.root / kind
        path = directory / f"{digest}.pt"
        meta_path = directory / f"{digest}.json"
        complete_path = directory / f"{digest}.complete.json"
        lock_path = directory / f"{digest}.lock"
        self._safe_directory(self.root, create=True)
        if self._type(directory) == "missing":
            directory.mkdir()
        self._safe_directory(directory, create=False)
        unit_paths = (path, meta_path, complete_path)
        try:
            present = self._inspect_unit(unit_paths)
            lock_kind = self._type(lock_path)
            if lock_kind not in {"missing", "file"}:
                raise ValueError(
                    f"unsafe cache writer lock ({lock_kind}): {lock_path}"
                )
        except ValueError:
            if self.resume:
                raise
            for unit_path in (*unit_paths, lock_path):
                self._remove_unsafe_unit(unit_path)
            present = (False, False, False)
            lock_kind = "missing"
        if lock_kind == "file" and not all(present):
            raise ValueError(f"cache unit has an active/stale writer lock: {lock_path}")
        if len(set(present)) != 1:
            if self.resume:
                raise ValueError(f"resume cache has incomplete unit: {path}")
            for unit_path in unit_paths:
                self._remove_unsafe_unit(unit_path)
        if all(present):
            try:
                complete = json.loads(complete_path.read_text())
                if complete != {
                    "schema_version": CACHE_SCHEMA_VERSION,
                    "key_sha256": stable_hash(canonical),
                }:
                    raise ValueError("cache complete marker mismatch")
                return self._load(path, meta_path, canonical, digest, kind)
            except Exception as exc:
                if self.resume:
                    raise ValueError(f"resume cache integrity check failed: {path}: {exc}") from exc
                for unit_path in unit_paths:
                    self._remove_unsafe_unit(unit_path)
        self._safe_directory(self.root, create=False)
        self._safe_directory(directory, create=False)
        self._inspect_unit(unit_paths)
        if self._type(lock_path) != "missing":
            raise ValueError(f"concurrent cache writer detected: {lock_path}")
        try:
            lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as exc:
            raise ValueError(f"concurrent cache writer detected: {lock_path}") from exc
        with os.fdopen(lock_fd, "w", encoding="utf-8") as stream:
            stream.write(json.dumps({"pid": os.getpid(), "key_sha256": stable_hash(canonical)}))
            stream.flush()
            os.fsync(stream.fileno())
        self._safe_unit_type(lock_path, allow_missing=False)
        self._safe_directory(directory, create=False)
        temporary: str | None = None
        try:
            values, solver_metadata = compute()
            values = values.detach().cpu()
            if not bool(torch.isfinite(values).all()):
                raise FloatingPointError("refusing to cache non-finite tensor")
            fd, temporary = tempfile.mkstemp(
                prefix=f".{digest}.", suffix=".pt", dir=directory
            )
            os.close(fd)
            torch.save({"schema_version": CACHE_SCHEMA_VERSION, "values": values,
                        "solver_metadata": solver_metadata}, temporary)
            with open(temporary, "rb") as stream:
                os.fsync(stream.fileno())
            os.replace(temporary, path)
            meta = {
                "schema_version": CACHE_SCHEMA_VERSION, "key": canonical,
                "key_sha256": stable_hash(canonical), "tensor_hash": tensor_hash(values),
                "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "shape": list(values.shape), "dtype": str(values.dtype),
                "solver_metadata": canonical_object(solver_metadata),
            }
            atomic_json(meta_path, meta)
            atomic_json(complete_path, {
                "schema_version": CACHE_SCHEMA_VERSION,
                "key_sha256": stable_hash(canonical),
            })
        finally:
            if temporary is not None and os.path.exists(temporary):
                os.unlink(temporary)
            if self._type(lock_path) == "file":
                lock_path.unlink()
            elif self._type(lock_path) != "missing":
                raise ValueError(f"cache writer lock changed type: {lock_path}")
        self.stats[kind]["misses"] += 1
        if kind == "states":
            self.stats[kind]["solver_invocations"] += 1
        return self._load(path, meta_path, canonical, digest, kind, count_hit=False)
