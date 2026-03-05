#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import itertools
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List


def parse_float_list(s: str) -> List[float]:
    vals = [x.strip() for x in s.split(",") if x.strip()]
    return [float(v) for v in vals]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Grid sweep for reservoir_burgers_1d.py over res-burgers-nu, feature-times, and Tr"
    )
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--script", default="reservoir_burgers_1d.py")
    p.add_argument("--out-root", default="visualizations/reservoir_burgers_nu_ft_sweep")

    p.add_argument("--nu-values", required=True, help="Comma-separated nu values")
    p.add_argument("--feature-times", required=True, help="Comma-separated feature-times values")
    p.add_argument(
        "--tr-values",
        default="0.2",
        help="Comma-separated Tr candidate values used to satisfy feature-time <= Tr",
    )

    p.add_argument("--data-file", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--elm-seed", type=int, default=0)
    p.add_argument("--ntrain", type=int, default=1000)
    p.add_argument("--ntest", type=int, default=200)
    p.add_argument("--sub", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=20)
    p.add_argument("--ridge-lambda", type=float, default=1e-4)
    p.add_argument("--elm-h", type=int, default=1024)
    p.add_argument("--device", default="auto")
    p.add_argument("--dt", type=float, default=0.01)
    p.add_argument("--burgers-fine-dt", type=float, default=1e-4)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--verbose", action="store_true")
    return p


def pick_tr(feature_time: float, tr_candidates: List[float]) -> float:
    feasible = [tr for tr in tr_candidates if tr + 1e-12 >= feature_time]
    if not feasible:
        raise ValueError(f"No feasible Tr for feature-time={feature_time}")
    return min(feasible)


def run_one(
    python: str,
    script: str,
    out_dir: Path,
    data_file: str,
    seed: int,
    elm_seed: int,
    ntrain: int,
    ntest: int,
    sub: int,
    batch_size: int,
    ridge_lambda: float,
    elm_h: int,
    device: str,
    dt: float,
    burgers_fine_dt: float,
    nu: float,
    feature_time: float,
    tr: float,
    dry_run: bool,
    verbose: bool,
) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        python,
        script,
        "--data-mode",
        "single_split",
        "--data-file",
        data_file,
        "--train-split",
        "0.8",
        "--shuffle",
        "--seed",
        str(seed),
        "--ntrain",
        str(ntrain),
        "--ntest",
        str(ntest),
        "--sub",
        str(sub),
        "--batch-size",
        str(batch_size),
        "--reservoir",
        "burgers",
        "--res-burgers-nu",
        str(nu),
        "--Tr",
        str(tr),
        "--dt",
        str(dt),
        "--K",
        "1",
        "--feature-times",
        str(feature_time),
        "--obs",
        "full",
        "--J",
        "1024",
        "--use-elm",
        "1",
        "--elm-h",
        str(elm_h),
        "--elm-activation",
        "tanh",
        "--elm-seed",
        str(elm_seed),
        "--ridge-lambda",
        str(ridge_lambda),
        "--ridge-dtype",
        "float64",
        "--standardize-features",
        "0",
        "--feature-std-eps",
        "1e-6",
        "--device",
        device,
        "--out-dir",
        str(out_dir),
        "--save-model",
        "--burgers-dealias",
        "--burgers-fine-dt",
        str(burgers_fine_dt),
        "--burgers-scheme",
        "split_step",
    ]
    if dry_run:
        cmd.append("--dry-run")
    if verbose:
        print("[run]", " ".join(cmd), flush=True)

    proc = subprocess.run(cmd, capture_output=True, text=True)
    (out_dir / "stdout_stderr.log").write_text(
        proc.stdout + "\n\n[stderr]\n" + proc.stderr, encoding="utf-8"
    )

    row: Dict[str, Any] = {
        "nu": nu,
        "feature_time": feature_time,
        "Tr": tr,
        "status": "ok" if proc.returncode == 0 else "fail",
        "return_code": proc.returncode,
        "train_relL2": None,
        "test_relL2": None,
        "elapsed_sec": None,
        "out_dir": str(out_dir),
    }

    cfg_path = out_dir / "run_config.json"
    if cfg_path.exists():
        try:
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            row["train_relL2"] = cfg.get("train_relL2")
            row["test_relL2"] = cfg.get("test_relL2")
            row["elapsed_sec"] = cfg.get("elapsed_sec")
        except Exception:
            pass
    return row


def main() -> None:
    args = build_parser().parse_args()

    nu_values = parse_float_list(args.nu_values)
    ft_values = parse_float_list(args.feature_times)
    tr_candidates = sorted(parse_float_list(args.tr_values))

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    combos = list(itertools.product(nu_values, ft_values))

    for idx, (nu, ft) in enumerate(combos):
        tr = pick_tr(ft, tr_candidates)
        trial_dir = out_root / f"trial_{idx:03d}_nu{nu:g}_ft{ft:g}_Tr{tr:g}"
        row = run_one(
            python=args.python,
            script=args.script,
            out_dir=trial_dir,
            data_file=args.data_file,
            seed=args.seed,
            elm_seed=args.elm_seed,
            ntrain=args.ntrain,
            ntest=args.ntest,
            sub=args.sub,
            batch_size=args.batch_size,
            ridge_lambda=args.ridge_lambda,
            elm_h=args.elm_h,
            device=args.device,
            dt=args.dt,
            burgers_fine_dt=args.burgers_fine_dt,
            nu=nu,
            feature_time=ft,
            tr=tr,
            dry_run=args.dry_run,
            verbose=args.verbose,
        )
        rows.append(row)
        print(
            f"[{idx + 1}/{len(combos)}] nu={nu:g} ft={ft:g} Tr={tr:g} -> "
            f"status={row['status']} test={row['test_relL2']}",
            flush=True,
        )

    summary_csv = out_root / "summary.csv"
    fields = [
        "nu",
        "feature_time",
        "Tr",
        "status",
        "return_code",
        "train_relL2",
        "test_relL2",
        "elapsed_sec",
        "out_dir",
    ]
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fields})

    ok_rows = [
        r for r in rows if r["status"] == "ok" and isinstance(r.get("test_relL2"), (int, float))
    ]
    ok_rows.sort(key=lambda r: float(r["test_relL2"]))

    ranking_json = out_root / "ranking.json"
    ranking_json.write_text(json.dumps(ok_rows, indent=2), encoding="utf-8")

    print("\nTop results (by test_relL2):")
    for i, r in enumerate(ok_rows[:10], start=1):
        print(
            f"{i:2d}. test={float(r['test_relL2']):.6g} train={float(r['train_relL2']):.6g} "
            f"nu={r['nu']} ft={r['feature_time']} Tr={r['Tr']}"
        )
    print(f"\nWrote: {summary_csv}")
    print(f"Wrote: {ranking_json}")


if __name__ == "__main__":
    main()
