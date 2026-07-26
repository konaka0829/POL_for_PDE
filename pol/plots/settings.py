"""Shared validation for generic plot task settings."""
from __future__ import annotations

from typing import Any, Mapping


def validate_formats_and_dpi(settings: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the common formats/dpi plot setting pair."""
    if set(settings) != {"formats", "dpi"}:
        raise ValueError("plot settings require formats and dpi")
    formats = settings["formats"]
    dpi = settings["dpi"]
    if (
        not isinstance(formats, list)
        or not formats
        or any(item not in {"png", "pdf", "svg"} for item in formats)
        or len(set(formats)) != len(formats)
    ):
        raise ValueError("formats must be a unique non-empty png/pdf/svg array")
    if isinstance(dpi, bool) or not isinstance(dpi, int) or dpi <= 0:
        raise ValueError("dpi must be a positive integer")
    return {"formats": list(formats), "dpi": dpi}
