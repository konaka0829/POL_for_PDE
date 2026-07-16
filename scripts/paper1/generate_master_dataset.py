#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pol.paper1.config import load_config_json
from pol.paper1.datasets import build_master_dataset, save_master_dataset
from pol.paper1.e0 import load_master_initial_conditions


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a Phase 1 Paper 1 master dataset")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-target", action="store_true", help="save only master initial conditions")
    parser.add_argument("--master-initial-conditions", help="validated E0 master_initial_conditions.pt archive")
    return parser


def _git_commit() -> str:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        if proc.returncode == 0:
            return proc.stdout.strip()
    except OSError:
        pass
    return "unknown"


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
        raise FileExistsError(f"{output_dir} already contains {names}; pass --overwrite")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    start = time.perf_counter()
    try:
        config = load_config_json(args.config)
    except Exception as exc:
        parser.error(f"invalid Paper 1 config: {exc}")

    out = Path(args.output_dir)
    try:
        _preflight_output_dir(out, overwrite=args.overwrite)
    except FileExistsError as exc:
        parser.error(str(exc))
    master = None if args.master_initial_conditions is None else load_master_initial_conditions(args.master_initial_conditions, config)
    dataset = build_master_dataset(config, generate_target=not args.no_target, master_initial_conditions=master)
    runtime = {
        "command": [sys.executable, str(Path(__file__).relative_to(REPO_ROOT)), *sys.argv[1:]],
        "git_commit": _git_commit(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "device": config.data.device,
        "dtype": config.data.dtype,
        "runtime_seconds": time.perf_counter() - start,
        "master_initial_conditions_archive": args.master_initial_conditions,
    }
    dataset.metadata["runtime"] = runtime
    save_master_dataset(dataset, out, overwrite=args.overwrite)
    shapes = {
        "sample_ids": list(dataset.sample_ids.shape),
        "u0_master": list(dataset.u0_master.shape),
        "u0_hat_master": list(dataset.u0_hat_master.shape),
        "y_target_master": None if dataset.y_target_master is None else list(dataset.y_target_master.shape),
    }
    print(json.dumps({"output_dir": str(out), "shapes": shapes}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
