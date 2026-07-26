"""Byte-stable JSON and hashing helpers for recipe orchestration."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any


def write_strict_json(path: Path, value: object) -> None:
    """Write sorted UTF-8 JSON with finite values and a trailing newline."""
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _e0_json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _e0_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_e0_json_safe(item) for item in value]
    try:
        import torch

        if isinstance(value, torch.Tensor):
            return _e0_json_safe(value.detach().cpu().tolist())
    except ImportError:
        pass
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_e0_json(path: Path, value: object) -> None:
    """Write E0 JSON using its tensor and non-finite sanitization contract."""
    write_strict_json(path, _e0_json_safe(value))


def file_sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file's bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()
