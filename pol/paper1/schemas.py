from __future__ import annotations

import hashlib
import json
from typing import Any


SCHEMA_VERSION = "paper1-master-dataset-v1"


def canonical_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def stable_hash_json(data: Any) -> str:
    return hashlib.sha256(canonical_json(data).encode("utf-8")).hexdigest()


def dataset_schema_metadata() -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "format": "torch_pt_with_manifest_json",
        "required_tensors": [
            "sample_ids",
            "train_indices",
            "val_indices",
            "test_indices",
            "u0_master",
            "u0_hat_master",
        ],
    }
