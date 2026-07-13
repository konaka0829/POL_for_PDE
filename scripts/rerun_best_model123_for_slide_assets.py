#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL123 = REPO_ROOT / "model123_burgers_1d.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rerun best Model123 sweep runs with saved model and predictions for slide assets.")
    parser.add_argument("--sweep-root", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--models", default="model1,model2,model3")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--device", default="")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def parse_models(raw: str) -> list[str]:
    models = [item.strip() for item in raw.split(",") if item.strip()]
    invalid = [model for model in models if model not in {"model1", "model2", "model3"}]
    if invalid:
        raise ValueError(f"Unsupported model(s): {invalid}")
    return models


def read_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Required file does not exist: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def best_row(path: Path, rank: int) -> dict[str, Any]:
    payload = read_json(path)
    rows = payload if isinstance(payload, list) else payload.get("best_runs", payload.get("rows"))
    if not isinstance(rows, list):
        raise ValueError(f"{path} must contain a list or a best_runs/rows list")
    if rank < 0 or rank >= len(rows):
        raise IndexError(f"--rank {rank} is out of range for {len(rows)} rows in {path}")
    row = rows[rank]
    if not isinstance(row, dict):
        raise ValueError(f"Best row at rank {rank} in {path} is not an object")
    return row


def command_from_run_config(run_config: dict[str, Any], python_exe: str) -> list[str]:
    command_line = run_config.get("command_line")
    if isinstance(command_line, list):
        cmd = [str(item) for item in command_line]
    elif isinstance(command_line, str) and command_line.strip():
        cmd = shlex.split(command_line)
    else:
        args = run_config.get("args", {})
        if not args:
            raise ValueError("run_config has neither command_line nor args fallback")
        cmd = [python_exe, str(MODEL123)]
        for key, value in args.items():
            if key.startswith("_") or value is None or key in {"out_dir", "save_model", "save_predictions"}:
                continue
            flag = "--" + key.replace("_", "-")
            if isinstance(value, bool):
                if value:
                    cmd.append(flag)
            else:
                cmd.extend([flag, str(value)])
    if len(cmd) < 2:
        raise ValueError(f"Could not reconstruct command from run_config: {cmd}")
    cmd[0] = python_exe
    if Path(cmd[1]).name == "model123_burgers_1d.py":
        cmd[1] = str(MODEL123)
    return cmd


def remove_flag(cmd: list[str], flag: str, takes_value: bool = True) -> list[str]:
    out: list[str] = []
    i = 0
    while i < len(cmd):
        if cmd[i] == flag:
            i += 2 if takes_value else 1
            continue
        out.append(cmd[i])
        i += 1
    return out


def remove_optional_value_flag(cmd: list[str], flag: str) -> list[str]:
    out: list[str] = []
    i = 0
    while i < len(cmd):
        if cmd[i] == flag:
            if i + 1 < len(cmd) and not cmd[i + 1].startswith("--"):
                i += 2
            else:
                i += 1
            continue
        out.append(cmd[i])
        i += 1
    return out


def rewrite_command(cmd: list[str], out_dir: Path, device: str, force: bool) -> list[str]:
    rewritten = list(cmd)
    for flag, takes_value in [
        ("--out-dir", True),
        ("--save-predictions", False),
        ("--device", True),
    ]:
        rewritten = remove_flag(rewritten, flag, takes_value=takes_value)
    rewritten = remove_optional_value_flag(rewritten, "--save-model")
    if out_dir.exists() and any(out_dir.iterdir()) and not force:
        raise FileExistsError(f"Output directory already exists and is non-empty: {out_dir}. Use --force to overwrite/run into it.")
    rewritten.extend(["--out-dir", str(out_dir), "--save-model", "--save-predictions"])
    if device:
        rewritten.extend(["--device", device])
    return rewritten


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    sweep_root = Path(args.sweep_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for model in parse_models(args.models):
        best_path = sweep_root / model / "best_runs.json"
        row = best_row(best_path, args.rank)
        source_run_dir = Path(row.get("run_dir", ""))
        if not source_run_dir.is_absolute():
            source_run_dir = (sweep_root / model / source_run_dir).resolve() if not (source_run_dir / "run_config.json").exists() else source_run_dir.resolve()
        run_config_path = source_run_dir / "run_config.json"
        run_config = read_json(run_config_path)
        out_dir = out_root / model
        cmd = rewrite_command(command_from_run_config(run_config, args.python), out_dir, args.device, args.force)
        log_path = out_dir / "stdout_stderr.log"
        out_dir.mkdir(parents=True, exist_ok=True)
        if args.dry_run:
            log_path.write_text("[dry-run]\n" + " ".join(shlex.quote(part) for part in cmd) + "\n", encoding="utf-8")
            returncode = 0
        else:
            proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
            log_path.write_text(proc.stdout + "\n\n[stderr]\n" + proc.stderr, encoding="utf-8")
            returncode = int(proc.returncode)
        rows.append(
            {
                "model": model,
                "rank": int(args.rank),
                "returncode": returncode,
                "source_run_dir": str(source_run_dir),
                "out_dir": str(out_dir),
                "command": " ".join(shlex.quote(part) for part in cmd),
                "log_path": str(log_path),
            }
        )
        if returncode != 0:
            break
    (out_root / "rerun_summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    write_csv(out_root / "rerun_summary.csv", rows)
    failures = [row for row in rows if row["returncode"] != 0]
    if failures:
        raise SystemExit(int(failures[0]["returncode"]))


if __name__ == "__main__":
    main()
