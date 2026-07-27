from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys


def _record_rejected_plot_request(
    raw: object, source: Path, repo_root: Path, failure: BaseException
) -> None:
    """Best-effort rejection record without changing a compute manifest."""
    if not isinstance(raw, dict):
        return
    run = raw.get("run")
    if not isinstance(run, dict):
        return
    name, output = run.get("name"), run.get("output_root")
    if (
        not isinstance(name, str)
        or not name
        or "/" in name
        or "\\" in name
        or ".." in name
        or not isinstance(output, str)
    ):
        return
    output_root = Path(output)
    if not output_root.is_absolute():
        output_root = repo_root / output_root
    run_dir = output_root.resolve() / name
    manifest_name = (
        "matrix_manifest.json"
        if raw.get("schema_version") == "paper1-matrix-run-v1"
        else "run_manifest.json"
    )
    manifest_path = run_dir / manifest_name
    if (
        run_dir.is_symlink()
        or manifest_path.is_symlink()
        or not manifest_path.is_file()
    ):
        return
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            not isinstance(manifest, dict)
            or manifest.get("run_name") != name
            or manifest.get("run_dir") != str(run_dir)
        ):
            return
        from .runtime.io import write_strict_json

        write_strict_json(
            run_dir / "resolved_plot_spec.json",
            {
                "schema_version": "paper1-plot-request-v1",
                "source_spec_path": str(source),
                "source_spec_sha256": hashlib.sha256(
                    source.read_bytes()
                ).hexdigest(),
                "compute_fingerprint": manifest.get(
                    "compute_fingerprint", manifest.get("science_fingerprint")
                ),
                "requested_tasks": raw.get("plots"),
                "request_mode": "plots_only",
                "requested_at": datetime.now(timezone.utc).isoformat(),
                "request_status": "rejected",
                "status": "rejected",
                "outcomes": [],
                "failure": f"{type(failure).__name__}: {failure}",
            },
        )
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return


def add_data_mode_args(
    parser: argparse.ArgumentParser,
    *,
    default_data_mode: str,
    default_data_file: str,
    default_train_file: str | None = None,
    default_test_file: str | None = None,
) -> argparse.ArgumentParser:
    parser.add_argument(
        "--data-mode",
        choices=("single_split", "separate_files"),
        default=default_data_mode,
        help="Data loading mode: single file split into train/test or separate files.",
    )
    parser.add_argument(
        "--data-file",
        default=default_data_file,
        help="Single data file used when --data-mode=single_split.",
    )
    parser.add_argument(
        "--train-file",
        default=default_train_file,
        help="Training data file used when --data-mode=separate_files.",
    )
    parser.add_argument(
        "--test-file",
        default=default_test_file,
        help="Test data file used when --data-mode=separate_files.",
    )
    return parser


def add_split_args(
    parser: argparse.ArgumentParser,
    *,
    default_train_split: float = 0.8,
    default_seed: int = 0,
) -> argparse.ArgumentParser:
    parser.add_argument(
        "--train-split",
        type=float,
        default=default_train_split,
        help="Fraction of samples to use for training in single_split mode.",
    )
    parser.add_argument("--seed", type=int, default=default_seed, help="Random seed.")
    parser.add_argument(
        "--shuffle",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Shuffle samples before splitting in single_split mode.",
    )
    return parser


def validate_data_mode_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.data_mode == "single_split":
        if not args.data_file:
            parser.error("--data-file is required when --data-mode=single_split")
        if hasattr(args, "train_split") and not (0.0 < args.train_split < 1.0):
            parser.error("--train-split must be in the interval (0, 1)")
    else:
        if not args.train_file or not args.test_file:
            parser.error("--train-file and --test-file are required when --data-mode=separate_files")


def _build_main_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pol", description="POL research commands")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="run a Paper 1 orchestration manifest")
    run.add_argument("run_spec")
    modes = run.add_mutually_exclusive_group()
    modes.add_argument("--plan", action="store_true", help="print the plan without side effects")
    modes.add_argument("--force", action="store_true", help="replace only this run directory")
    modes.add_argument(
        "--plots-only",
        action="store_true",
        help="verify compute artifacts and execute only plot tasks",
    )
    verify = commands.add_parser(
        "verify", help="verify a completed Paper 1 run"
    )
    verify.add_argument("run_dir")
    verify.add_argument(
        "--deep",
        action="store_true",
        help="recompute E0 numerical validation in addition to artifact QA",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the package-level command line interface."""
    parser = _build_main_parser()
    args = parser.parse_args(argv)
    repo_root = Path(__file__).resolve().parent.parent
    source = Path()
    raw: object = None
    try:
        if args.command == "verify":
            run_dir = Path(args.run_dir).resolve()
            from .paper1.artifact_contracts import E0ArtifactContract

            E0ArtifactContract().validate_complete(run_dir / "e0")
            if args.deep:
                from .paper1.e0_validation import (
                    deep_validate_e0_scientific_artifacts,
                )

                deep_validate_e0_scientific_artifacts(run_dir / "e0")
            print(
                json.dumps(
                    {
                        "status": "pass",
                        "run_dir": str(run_dir),
                        "mode": "deep" if args.deep else "artifact_only",
                    },
                    sort_keys=True,
                )
            )
            return 0
        source = Path(args.run_spec).resolve()
        raw = json.loads(source.read_text(encoding="utf-8"))
        schema = raw.get("schema_version") if isinstance(raw, dict) else None
        if schema == "paper1-matrix-run-v1":
            from .workflow.matrix import execute_matrix_run, matrix_plan_to_dict
            from .workflow.matrix_spec import load_matrix_spec

            matrix_spec = load_matrix_spec(source, repo_root=repo_root)
            if args.plan:
                print(
                    json.dumps(
                        matrix_plan_to_dict(matrix_spec, repo_root=repo_root),
                        indent=2,
                        allow_nan=False,
                    )
                )
                return 0
            return execute_matrix_run(
                matrix_spec,
                repo_root=repo_root,
                force=args.force,
                plots_only=args.plots_only,
            )
        from .paper1.run_spec import load_run_spec
        from .paper1.runner import execute_run, plan_to_dict

        spec = load_run_spec(source, repo_root=repo_root)
        if args.plan:
            print(
                json.dumps(
                    plan_to_dict(spec, repo_root=repo_root),
                    indent=2,
                    allow_nan=False,
                )
            )
            return 0
        return execute_run(
            spec,
            repo_root=repo_root,
            force=args.force,
            plots_only=args.plots_only,
        )
    except (OSError, ValueError) as exc:
        if getattr(args, "plots_only", False):
            _record_rejected_plot_request(raw, source, repo_root, exc)
        print(f"pol: error: {exc}", file=sys.stderr)
        return 2
