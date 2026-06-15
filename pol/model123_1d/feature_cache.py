from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from pol.cache import stable_hash, to_jsonable


FEATURE_CODE_VERSION = "feature_cache_v1"


def make_feature_cache_key(
    *,
    dataset_hash: str,
    split_hash: str,
    surrogate_config: dict[str, Any],
    observation_config: dict[str, Any],
) -> dict[str, str]:
    surrogate_hash = stable_hash(surrogate_config)
    observation_hash = stable_hash(observation_config)
    return {
        "dataset_hash": dataset_hash,
        "split_hash": split_hash,
        "surrogate_hash": surrogate_hash,
        "observation_hash": observation_hash,
        "feature_code_version": FEATURE_CODE_VERSION,
    }


def feature_cache_dir(root: str | Path, key: dict[str, str]) -> Path:
    return (
        Path(root)
        / f"dataset_{key['dataset_hash'][:16]}"
        / f"split_{key['split_hash'][:16]}"
        / f"surrogate_{key['surrogate_hash'][:16]}"
        / f"obs_{key['observation_hash'][:16]}"
    )


def save_feature_cache(path: str | Path, *, tensors: dict[str, torch.Tensor], metadata: dict[str, Any]) -> None:
    root = Path(path)
    root.mkdir(parents=True, exist_ok=True)
    for name, tensor in tensors.items():
        torch.save(tensor.detach().cpu(), root / f"{name}_features.pt")
    (root / "metadata.json").write_text(json.dumps(to_jsonable(metadata), indent=2), encoding="utf-8")


def load_feature_cache(path: str | Path, *, expected_metadata: dict[str, Any] | None = None) -> dict[str, torch.Tensor]:
    root = Path(path)
    meta_path = root / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"feature cache metadata not found: {meta_path}")
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    if expected_metadata:
        for key, expected in expected_metadata.items():
            if metadata.get(key) != expected:
                raise ValueError(f"feature cache metadata mismatch for {key}: expected {expected!r}, found {metadata.get(key)!r}")
    tensors: dict[str, torch.Tensor] = {}
    for split in ("train", "val", "test"):
        tensor_path = root / f"{split}_features.pt"
        if tensor_path.exists():
            tensors[split] = torch.load(tensor_path, map_location="cpu")
    return tensors
