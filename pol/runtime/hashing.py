"""Canonical content identities shared by recipes and workflows."""
from __future__ import annotations

import hashlib
from typing import Any

import torch

from .io import canonical_json_bytes, file_sha256


def stable_object_hash(value: object) -> str:
    """Hash a JSON-compatible object independently of formatting."""
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def tensor_sha256(value: torch.Tensor) -> str:
    """Hash tensor dtype, shape, and contiguous CPU bytes."""
    tensor = value.detach().cpu().contiguous()
    header: dict[str, Any] = {
        "dtype": str(tensor.dtype),
        "shape": list(tensor.shape),
    }
    digest = hashlib.sha256(canonical_json_bytes(header))
    digest.update(tensor.view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


__all__ = ["file_sha256", "stable_object_hash", "tensor_sha256"]
