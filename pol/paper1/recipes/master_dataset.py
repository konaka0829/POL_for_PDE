"""Import-safe Paper 1 master-dataset artifact orchestration."""
from __future__ import annotations

import hashlib
from pathlib import Path
import platform
import time

import torch

from pol.runtime.provenance import git_output
from pol.runtime.recipe import RecipeInvocation, RecipeResult, RecipeUsageError

def _load_science_dependencies() -> None:
    """Load Paper 1 scientific modules only when the recipe is invoked."""
    global build_master_dataset, load_config_json
    global load_master_initial_conditions, save_master_dataset

    from ..config import load_config_json
    from ..datasets import (
        build_master_dataset,
        save_master_dataset,
    )
    from ..e0 import load_master_initial_conditions


def _preflight_output_dir(output_dir: Path, *, overwrite: bool) -> None:
    existing = [
        path
        for path in (
            output_dir / "master_dataset.pt",
            output_dir / "manifest.json",
            output_dir / "resolved_config.json",
        )
        if path.exists()
    ]
    if existing and not overwrite:
        names = ", ".join(path.name for path in existing)
        raise RecipeUsageError(
            f"{output_dir} already contains {names}; pass --overwrite"
        )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_e0_record(
    e0_dir: Path, master_tensor_hash: str
) -> dict[str, object]:
    source_files = (
        "e0_summary.json",
        "accepted_production_config.json",
        "master_initial_conditions.pt",
        "master_manifest.json",
    )
    missing = [name for name in source_files if not (e0_dir / name).is_file()]
    if missing:
        raise ValueError("E0 provenance artifact is missing: " + missing[0])
    return {
        "schema_version": "paper1-dataset-source-e0-v1",
        "files": {name: _sha256(e0_dir / name) for name in source_files},
        "master_tensor_hash": master_tensor_hash,
    }


def run_master_dataset_generation(
    config_path: Path,
    output_dir: Path,
    *,
    overwrite: bool,
    generate_target: bool,
    master_initial_conditions: Path | None,
    invocation: RecipeInvocation,
) -> RecipeResult:
    """Generate the master dataset without CLI dependencies."""
    _load_science_dependencies()
    start = time.perf_counter()
    try:
        config = load_config_json(config_path)
    except Exception as exc:
        raise RecipeUsageError(f"invalid Paper 1 config: {exc}") from exc
    _preflight_output_dir(output_dir, overwrite=overwrite)

    master = (
        None
        if master_initial_conditions is None
        else load_master_initial_conditions(master_initial_conditions, config)
    )
    dataset = build_master_dataset(
        config,
        generate_target=generate_target,
        master_initial_conditions=master,
    )
    if master_initial_conditions is not None:
        archive = master_initial_conditions
        e0_dir = archive.parent
        if all(
            (e0_dir / name).is_file()
            for name in (
                "e0_summary.json",
                "accepted_production_config.json",
                "master_initial_conditions.pt",
                "master_manifest.json",
            )
        ):
            dataset.metadata["source_e0"] = _source_e0_record(
                e0_dir, dataset.metadata["tensor_hashes"]["u0_master"]
            )
    runtime = {
        "command": list(invocation.command),
        "git_commit": git_output(invocation.repo_root, ["rev-parse", "HEAD"]),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "device": config.data.device,
        "dtype": config.data.dtype,
        "runtime_seconds": time.perf_counter() - start,
        "master_initial_conditions_archive": (
            None
            if master_initial_conditions is None
            else str(master_initial_conditions)
        ),
    }
    dataset.metadata["runtime"] = runtime
    save_master_dataset(dataset, output_dir, overwrite=overwrite)
    shapes = {
        "sample_ids": list(dataset.sample_ids.shape),
        "u0_master": list(dataset.u0_master.shape),
        "u0_hat_master": list(dataset.u0_hat_master.shape),
        "y_target_master": (
            None
            if dataset.y_target_master is None
            else list(dataset.y_target_master.shape)
        ),
    }
    payload = {"output_dir": str(output_dir), "shapes": shapes}
    return RecipeResult(
        status="pass",
        exit_code=0,
        output_dir=output_dir,
        console_payload=payload,
    )
