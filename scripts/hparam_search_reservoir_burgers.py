#!/usr/bin/env python3
import argparse
import csv
import json
import math
import random
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


@dataclass
class Trial:
    stage: str
    trial_id: str
    parent_id: str
    rd_nu: float
    rd_beta: float
    rd_alpha: float
    dt: float
    K: int
    ridge_lambda: float
    seed: int
    elm_seed: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Staged hyperparameter search for reservoir_burgers_1d.py"
    )
    parser.add_argument("--python", default=sys.executable, help="Python executable")
    parser.add_argument("--script", default="reservoir_burgers_1d.py", help="Target training script")
    parser.add_argument(
        "--out-root",
        default="visualizations/reservoir_burgers_hsearch",
        help="Root directory for all search outputs",
    )
    parser.add_argument("--search-seed", type=int, default=42, help="Search RNG seed")

    # Stage budgets
    parser.add_argument(
        "--coarse-trials",
        "--stage-a-trials",
        dest="stage_a_trials",
        type=int,
        default=48,
        help="Stage A: number of coarse random trials for PDE reservoir params.",
    )
    parser.add_argument(
        "--coarse-topk",
        "--stage-b-top",
        dest="stage_b_top",
        type=int,
        default=8,
        help="Stage B: number of best Stage A trials kept as refinement parents.",
    )
    parser.add_argument(
        "--local-trials-per-parent",
        "--stage-b-local",
        dest="stage_b_local",
        type=int,
        default=6,
        help="Stage B: local perturbation trials generated per parent config.",
    )
    parser.add_argument(
        "--refine-topk",
        "--stage-c-top",
        dest="stage_c_top",
        type=int,
        default=3,
        help="Stage C: number of best Stage B trials kept for final refinement.",
    )
    parser.add_argument(
        "--refine-trials-per-parent",
        "--stage-c-per-top",
        dest="stage_c_per_top",
        type=int,
        default=12,
        help="Stage C: trials per parent tuning dt/K/ridge + nearby PDE params.",
    )

    # Final robust check
    parser.add_argument(
        "--final-candidates",
        "--final-top",
        dest="final_top",
        type=int,
        default=3,
        help="Number of top Stage C candidates sent to multi-seed robustness check.",
    )
    parser.add_argument(
        "--robustness-seeds",
        "--final-seeds",
        dest="final_seeds",
        type=str,
        default="0,1,2,3,4",
        help="Comma-separated seeds for final robustness runs",
    )

    # Base command (from your reference run)
    parser.add_argument("--data-mode", default="single_split")
    parser.add_argument("--data-file", default="data/burgers_data_R10.mat")
    parser.add_argument("--train-split", type=float, default=0.8)
    parser.add_argument("--shuffle", action="store_true", default=True)
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--ntrain", type=int, default=1000)
    parser.add_argument("--ntest", type=int, default=200)
    parser.add_argument("--sub", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=20)

    parser.add_argument("--reservoir", default="reaction_diffusion")
    parser.add_argument("--Tr", type=float, default=1.0)
    parser.add_argument("--obs", default="full")
    parser.add_argument("--J", type=int, default=1024)

    parser.add_argument("--use-elm", type=int, default=1)
    parser.add_argument("--elm-h", type=int, default=4096)
    parser.add_argument("--elm-activation", default="tanh")
    parser.add_argument("--base-elm-seed", type=int, default=0)

    parser.add_argument("--ridge-dtype", default="float64")
    parser.add_argument("--standardize-features", type=int, default=0)
    parser.add_argument("--feature-std-eps", type=float, default=1e-6)
    parser.add_argument("--device", default="auto")

    parser.add_argument("--save-model", action="store_true")
    parser.add_argument(
        "--target-dry-run",
        action="store_true",
        help="Pass --dry-run to reservoir_burgers_1d.py for smoke checks.",
    )
    parser.add_argument("--verbose", action="store_true")

    return parser.parse_args()


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def sample_log_uniform(rng: random.Random, lo: float, hi: float) -> float:
    return 10.0 ** rng.uniform(math.log10(lo), math.log10(hi))


def run_trial(args: argparse.Namespace, trial: Trial, root: Path) -> dict[str, Any]:
    out_dir = root / trial.stage / trial.trial_id
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        args.python,
        args.script,
        "--data-mode",
        args.data_mode,
        "--data-file",
        args.data_file,
        "--train-split",
        str(args.train_split),
        "--seed",
        str(trial.seed),
        "--ntrain",
        str(args.ntrain),
        "--ntest",
        str(args.ntest),
        "--sub",
        str(args.sub),
        "--batch-size",
        str(args.batch_size),
        "--reservoir",
        args.reservoir,
        "--Tr",
        str(args.Tr),
        "--dt",
        str(trial.dt),
        "--K",
        str(trial.K),
        "--obs",
        args.obs,
        "--J",
        str(args.J),
        "--use-elm",
        str(args.use_elm),
        "--elm-h",
        str(args.elm_h),
        "--elm-activation",
        args.elm_activation,
        "--elm-seed",
        str(trial.elm_seed),
        "--ridge-lambda",
        str(trial.ridge_lambda),
        "--ridge-dtype",
        args.ridge_dtype,
        "--rd-nu",
        str(trial.rd_nu),
        "--rd-beta",
        str(trial.rd_beta),
        "--rd-alpha",
        str(trial.rd_alpha),
        "--standardize-features",
        str(args.standardize_features),
        "--feature-std-eps",
        str(args.feature_std_eps),
        "--device",
        args.device,
        "--out-dir",
        str(out_dir),
    ]

    if args.shuffle:
        cmd.append("--shuffle")
    if args.save_model:
        cmd.append("--save-model")
    if args.target_dry_run:
        cmd.append("--dry-run")

    if args.verbose:
        print("[run]", " ".join(cmd), flush=True)

    proc = subprocess.run(cmd, capture_output=True, text=True)

    result = {
        "stage": trial.stage,
        "trial_id": trial.trial_id,
        "parent_id": trial.parent_id,
        "rd_nu": trial.rd_nu,
        "rd_beta": trial.rd_beta,
        "rd_alpha": trial.rd_alpha,
        "dt": trial.dt,
        "K": trial.K,
        "ridge_lambda": trial.ridge_lambda,
        "seed": trial.seed,
        "elm_seed": trial.elm_seed,
        "status": "ok" if proc.returncode == 0 else "fail",
        "return_code": proc.returncode,
        "out_dir": str(out_dir),
        "train_relL2": None,
        "test_relL2": None,
        "elapsed_sec": None,
    }

    log_path = out_dir / "stdout_stderr.log"
    log_path.write_text(proc.stdout + "\n\n[stderr]\n" + proc.stderr, encoding="utf-8")

    cfg_path = out_dir / "run_config.json"
    if cfg_path.exists():
        try:
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            result["train_relL2"] = cfg.get("train_relL2")
            result["test_relL2"] = cfg.get("test_relL2")
            result["elapsed_sec"] = cfg.get("elapsed_sec")
        except Exception:
            pass

    return result


def top_ok(results: list[dict[str, Any]], stage: str, n: int) -> list[dict[str, Any]]:
    rows = [
        r
        for r in results
        if r["stage"] == stage and r["status"] == "ok" and isinstance(r["test_relL2"], (int, float))
    ]
    rows.sort(key=lambda r: r["test_relL2"])
    return rows[:n]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fields = [
        "stage",
        "trial_id",
        "parent_id",
        "rd_nu",
        "rd_beta",
        "rd_alpha",
        "dt",
        "K",
        "ridge_lambda",
        "seed",
        "elm_seed",
        "status",
        "return_code",
        "train_relL2",
        "test_relL2",
        "elapsed_sec",
        "out_dir",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fields})


def make_stage_a(args: argparse.Namespace, rng: random.Random) -> list[Trial]:
    trials: list[Trial] = []
    for i in range(args.stage_a_trials):
        nu = sample_log_uniform(rng, 3e-5, 1e-2)
        alpha = sample_log_uniform(rng, 0.3, 3.0)
        beta = rng.uniform(-1.0, 1.0)
        trials.append(
            Trial(
                stage="A",
                trial_id=f"A_{i:04d}",
                parent_id="-",
                rd_nu=nu,
                rd_beta=beta,
                rd_alpha=alpha,
                dt=0.01,
                K=2,
                ridge_lambda=1e-4,
                seed=args.base_seed,
                elm_seed=args.base_elm_seed,
            )
        )
    return trials


def make_stage_b(args: argparse.Namespace, rng: random.Random, parents: list[dict[str, Any]]) -> list[Trial]:
    trials: list[Trial] = []
    for p in parents:
        parent_id = p["trial_id"]
        center = Trial(
            stage="B",
            trial_id=f"B_{parent_id}_center",
            parent_id=parent_id,
            rd_nu=float(p["rd_nu"]),
            rd_beta=float(p["rd_beta"]),
            rd_alpha=float(p["rd_alpha"]),
            dt=0.01,
            K=2,
            ridge_lambda=1e-4,
            seed=args.base_seed,
            elm_seed=args.base_elm_seed,
        )
        trials.append(center)

        for j in range(args.stage_b_local):
            d_nu = rng.uniform(-0.4, 0.4)
            d_alpha = rng.uniform(-0.3, 0.3)
            nu = clamp(center.rd_nu * (10.0 ** d_nu), 3e-5, 1e-2)
            alpha = clamp(center.rd_alpha * (10.0 ** d_alpha), 0.2, 4.0)
            beta = clamp(center.rd_beta + rng.uniform(-0.35, 0.35), -1.5, 1.5)
            trials.append(
                Trial(
                    stage="B",
                    trial_id=f"B_{parent_id}_{j:03d}",
                    parent_id=parent_id,
                    rd_nu=nu,
                    rd_beta=beta,
                    rd_alpha=alpha,
                    dt=0.01,
                    K=2,
                    ridge_lambda=1e-4,
                    seed=args.base_seed,
                    elm_seed=args.base_elm_seed,
                )
            )
    return trials


def make_stage_c(args: argparse.Namespace, rng: random.Random, parents: list[dict[str, Any]]) -> list[Trial]:
    trials: list[Trial] = []
    dt_choices = [0.005, 0.01, 0.02]
    k_choices = [2, 3, 4]

    for p in parents:
        parent_id = p["trial_id"]
        center = Trial(
            stage="C",
            trial_id=f"C_{parent_id}_center",
            parent_id=parent_id,
            rd_nu=float(p["rd_nu"]),
            rd_beta=float(p["rd_beta"]),
            rd_alpha=float(p["rd_alpha"]),
            dt=float(p["dt"]),
            K=int(p["K"]),
            ridge_lambda=float(p["ridge_lambda"]),
            seed=args.base_seed,
            elm_seed=args.base_elm_seed,
        )
        trials.append(center)

        for j in range(args.stage_c_per_top):
            nu = clamp(center.rd_nu * (10.0 ** rng.uniform(-0.25, 0.25)), 3e-5, 1e-2)
            alpha = clamp(center.rd_alpha * (10.0 ** rng.uniform(-0.2, 0.2)), 0.2, 4.0)
            beta = clamp(center.rd_beta + rng.uniform(-0.2, 0.2), -1.5, 1.5)
            ridge = sample_log_uniform(rng, 1e-6, 1e-2)

            trials.append(
                Trial(
                    stage="C",
                    trial_id=f"C_{parent_id}_{j:03d}",
                    parent_id=parent_id,
                    rd_nu=nu,
                    rd_beta=beta,
                    rd_alpha=alpha,
                    dt=rng.choice(dt_choices),
                    K=rng.choice(k_choices),
                    ridge_lambda=ridge,
                    seed=args.base_seed,
                    elm_seed=args.base_elm_seed,
                )
            )
    return trials


def make_final(args: argparse.Namespace, parents: list[dict[str, Any]]) -> list[Trial]:
    seed_list = [int(x.strip()) for x in args.final_seeds.split(",") if x.strip()]
    trials: list[Trial] = []
    for p in parents:
        parent_id = p["trial_id"]
        for s in seed_list:
            trials.append(
                Trial(
                    stage="FINAL",
                    trial_id=f"F_{parent_id}_seed{s}",
                    parent_id=parent_id,
                    rd_nu=float(p["rd_nu"]),
                    rd_beta=float(p["rd_beta"]),
                    rd_alpha=float(p["rd_alpha"]),
                    dt=float(p["dt"]),
                    K=int(p["K"]),
                    ridge_lambda=float(p["ridge_lambda"]),
                    seed=s,
                    elm_seed=s,
                )
            )
    return trials


def aggregate_final(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    finals = [
        r
        for r in rows
        if r["stage"] == "FINAL" and r["status"] == "ok" and isinstance(r["test_relL2"], (int, float))
    ]

    by_parent: dict[str, list[dict[str, Any]]] = {}
    for r in finals:
        by_parent.setdefault(str(r["parent_id"]), []).append(r)

    summary: list[dict[str, Any]] = []
    for parent, group in by_parent.items():
        best = min(group, key=lambda x: x["test_relL2"])
        scores = [float(g["test_relL2"]) for g in group]
        summary.append(
            {
                "parent_id": parent,
                "n_runs": len(group),
                "mean_test_relL2": mean(scores),
                "std_test_relL2": pstdev(scores) if len(scores) > 1 else 0.0,
                "robust_score": mean(scores) + 0.5 * (pstdev(scores) if len(scores) > 1 else 0.0),
                "best_test_relL2": best["test_relL2"],
                "rd_nu": best["rd_nu"],
                "rd_beta": best["rd_beta"],
                "rd_alpha": best["rd_alpha"],
                "dt": best["dt"],
                "K": best["K"],
                "ridge_lambda": best["ridge_lambda"],
            }
        )

    summary.sort(key=lambda x: x["robust_score"])
    return summary


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "parent_id",
        "n_runs",
        "mean_test_relL2",
        "std_test_relL2",
        "robust_score",
        "best_test_relL2",
        "rd_nu",
        "rd_beta",
        "rd_alpha",
        "dt",
        "K",
        "ridge_lambda",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fields})


def main() -> None:
    args = parse_args()
    rng = random.Random(args.search_seed)

    root = Path(args.out_root)
    root.mkdir(parents=True, exist_ok=True)

    all_results: list[dict[str, Any]] = []

    print("== Stage A: coarse PDE search ==", flush=True)
    trials_a = make_stage_a(args, rng)
    for t in trials_a:
        all_results.append(run_trial(args, t, root))
    best_a = top_ok(all_results, "A", args.stage_b_top)
    print(f"Stage A complete: ok={len([r for r in all_results if r['stage']=='A' and r['status']=='ok'])}, top={len(best_a)}", flush=True)

    print("== Stage B: local PDE refinement ==", flush=True)
    trials_b = make_stage_b(args, rng, best_a)
    for t in trials_b:
        all_results.append(run_trial(args, t, root))
    best_b = top_ok(all_results, "B", args.stage_c_top)
    print(f"Stage B complete: ok={len([r for r in all_results if r['stage']=='B' and r['status']=='ok'])}, top={len(best_b)}", flush=True)

    print("== Stage C: tune dt/K/ridge near best PDE ==", flush=True)
    trials_c = make_stage_c(args, rng, best_b)
    for t in trials_c:
        all_results.append(run_trial(args, t, root))
    best_c = top_ok(all_results, "C", args.final_top)
    print(f"Stage C complete: ok={len([r for r in all_results if r['stage']=='C' and r['status']=='ok'])}, finalists={len(best_c)}", flush=True)

    print("== Final: multi-seed robustness check ==", flush=True)
    trials_f = make_final(args, best_c)
    for t in trials_f:
        all_results.append(run_trial(args, t, root))

    summary = aggregate_final(all_results)

    write_csv(root / "results.csv", all_results)
    write_summary_csv(root / "summary.csv", summary)
    (root / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Saved results: {root / 'results.csv'}", flush=True)
    print(f"Saved summary: {root / 'summary.csv'}", flush=True)

    if summary:
        top = summary[0]
        print("Best robust config:", flush=True)
        print(
            json.dumps(
                {
                    "rd_nu": top["rd_nu"],
                    "rd_beta": top["rd_beta"],
                    "rd_alpha": top["rd_alpha"],
                    "dt": top["dt"],
                    "K": top["K"],
                    "ridge_lambda": top["ridge_lambda"],
                    "mean_test_relL2": top["mean_test_relL2"],
                    "std_test_relL2": top["std_test_relL2"],
                    "robust_score": top["robust_score"],
                },
                indent=2,
                ensure_ascii=False,
            ),
            flush=True,
        )
    else:
        print("No successful FINAL runs. Check logs under stage directories.", flush=True)


if __name__ == "__main__":
    main()
