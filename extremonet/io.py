from __future__ import annotations

from typing import Any

import torch

from extremonet.model import ExtremONet


def save_eon(
    model: ExtremONet,
    path: str,
    *,
    extra_meta: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
) -> None:
    payload = {
        "class_name": model.__class__.__name__,
        "config": model.get_config() if config is None else config,
        "state_dict": model.state_dict(),
        "extra_meta": {} if extra_meta is None else extra_meta,
    }
    torch.save(payload, path)


def load_eon(path: str, map_location: str | torch.device = "cpu") -> tuple[ExtremONet, dict[str, Any]]:
    payload = torch.load(path, map_location=map_location)
    if "config" not in payload or "state_dict" not in payload:
        raise ValueError("Invalid EON checkpoint: missing config/state_dict")

    model = ExtremONet.from_config(payload["config"])
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, payload.get("extra_meta", {})
