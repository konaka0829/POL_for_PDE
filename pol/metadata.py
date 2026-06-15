from __future__ import annotations

import datetime as _dt
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

try:
    import scipy
except Exception:  # pragma: no cover - optional runtime metadata
    scipy = None

try:
    import torch
except Exception:  # pragma: no cover - optional runtime metadata
    torch = None


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, _dt.datetime):
        return value.isoformat()
    if torch is not None and isinstance(value, torch.dtype):
        return str(value).replace("torch.", "")
    if torch is not None and isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return to_jsonable(value.detach().cpu().item())
        return {"shape": list(value.shape), "dtype": str(value.dtype).replace("torch.", "")}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    return value


def stable_json_dumps(payload: Any) -> str:
    return json.dumps(to_jsonable(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def stable_json_hash(payload: Any) -> str:
    return hashlib.sha256(stable_json_dumps(payload).encode("utf-8")).hexdigest()


def file_sha256(path: str | Path | None) -> str | None:
    if not path:
        return None
    p = Path(path)
    if not p.exists() or not p.is_file():
        return None
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def get_git_info(repo_root: str | Path) -> dict[str, Any]:
    root = Path(repo_root)
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"],
                cwd=root,
                text=True,
                stderr=subprocess.STDOUT,
            ).strip()
        )
        return {"git_commit": commit, "git_dirty": dirty, "git_error": None}
    except Exception as exc:  # pragma: no cover - depends on environment
        return {"git_commit": None, "git_dirty": None, "git_error": str(exc)}


def get_runtime_info() -> dict[str, Any]:
    payload = {
        "created_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "scipy_version": getattr(scipy, "__version__", None) if scipy is not None else None,
    }
    if torch is not None:
        payload["torch_version"] = torch.__version__
    return payload


def get_command_line() -> list[str]:
    return list(sys.argv)


_ALIASES = {
    "nu": "target_nu",
    "target_nu": "target_nu",
    "L": "domain_length",
    "length": "domain_length",
    "domain_length": "domain_length",
    "time_step": "dt",
    "dt": "dt",
    "final_time": "T",
    "T": "T",
    "nx": "nx",
    "ic_type": "ic_type",
    "solver": "solver",
    "scheme": "solver",
    "time_integrator": "time_integrator",
    "burgers_scheme": "burgers_scheme",
    "dealias": "dealias",
    "equation": "target_equation",
    "target_equation": "target_equation",
}


def normalize_meta_value(value: Any) -> Any:
    if torch is not None and isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return None
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return None
        if value.dtype.kind in {"U", "S", "O"}:
            flat = [normalize_meta_value(item) for item in value.reshape(-1)]
            flat = [item for item in flat if item is not None]
            if not flat:
                return None
            if all(isinstance(item, str) and len(item) == 1 for item in flat):
                return "".join(flat)
            if len(flat) == 1:
                return flat[0]
            return flat
        else:
            if value.size > 1:
                return [normalize_meta_value(item) for item in value.reshape(-1)]
            value = value.reshape(-1)[0].item()
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.bytes_):
        return value.astype(str).item()
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        flat = [normalize_meta_value(item) for item in value]
        flat = [item for item in flat if item is not None]
        if len(flat) == 1:
            return flat[0]
        return flat
    if isinstance(value, np.generic):
        return value.item()
    return value


def _metadata_scalar(value: Any) -> Any:
    value = normalize_meta_value(value)
    if isinstance(value, list):
        return value[0] if value else None
    return value


def normalize_dataset_metadata(raw_metadata: dict[str, Any] | None) -> dict[str, Any]:
    raw = {} if raw_metadata is None else dict(raw_metadata)
    normalized: dict[str, Any] = {}
    for key, value in raw.items():
        if key.startswith("__"):
            continue
        canonical = _ALIASES.get(key, key)
        scalar = _metadata_scalar(value)
        if scalar is None:
            continue
        normalized[canonical] = scalar
    if "target_nu" not in normalized and "nu" in raw:
        normalized["target_nu"] = _metadata_scalar(raw["nu"])
    normalized["raw_metadata"] = to_jsonable(raw)
    return normalized


def _compare_metadata_value(expected: Any, found: Any) -> bool:
    if expected is None or found is None:
        return expected is None and found is None
    if isinstance(expected, bool) or isinstance(found, bool):
        return bool(expected) == bool(found)
    if isinstance(expected, (int, float, np.integer, np.floating)):
        try:
            return bool(np.isclose(float(expected), float(found), rtol=1e-9, atol=1e-12))
        except Exception:
            return False
    return str(expected) == str(found)


def validate_dataset_metadata(
    *,
    raw_metadata: dict[str, Any] | None,
    expected: dict[str, Any],
    strict: bool = True,
    require_complete_metadata: bool = False,
) -> dict[str, Any]:
    normalized = normalize_dataset_metadata(raw_metadata)
    checks: dict[str, dict[str, Any]] = {}
    warnings: list[str] = []
    for key, exp in expected.items():
        if exp is None:
            continue
        found = normalized.get(key)
        if found is None:
            warnings.append(f"Dataset metadata missing {key}; expected {exp!r}.")
            checks[key] = {"expected": exp, "found": None, "ok": False, "missing": True}
            continue
        ok = _compare_metadata_value(exp, found)
        checks[key] = {"expected": exp, "found": found, "ok": ok, "missing": False}
        if key == "nx":
            checks[key]["meaning"] = "dataset/raw spatial resolution before --sub"
    mismatches = [key for key, check in checks.items() if not check["ok"] and not check.get("missing")]
    missing = [key for key, check in checks.items() if check.get("missing")]
    ok = not mismatches and (not missing or not require_complete_metadata)
    message_parts = []
    if mismatches:
        message_parts.extend(
            f"{key}: expected {checks[key]['expected']!r}, found {checks[key]['found']!r}"
            for key in mismatches
        )
    result = {
        "strict": bool(strict),
        "strict_mismatch": bool(strict),
        "require_complete_metadata": bool(require_complete_metadata),
        "ok": bool(ok),
        "has_missing": bool(missing),
        "has_mismatch": bool(mismatches),
        "missing": missing,
        "mismatches": mismatches,
        "checks": to_jsonable(checks),
        "warnings": warnings,
        "normalized_metadata": to_jsonable(normalized),
        "raw_metadata": to_jsonable(raw_metadata or {}),
    }
    if strict and mismatches:
        raise ValueError("Dataset metadata mismatch: " + "; ".join(message_parts))
    if require_complete_metadata and missing:
        raise ValueError(
            "Dataset metadata incomplete: "
            + "; ".join(f"{key}: expected {checks[key]['expected']!r}, found missing" for key in missing)
        )
    return result
